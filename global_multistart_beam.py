"""Build new ESA itineraries with a bounded, multi-start beam."""

from collections import Counter
from functools import lru_cache
import math
from pathlib import Path
import subprocess
import sys
import time

from anchored_two_stop_search import INCUMBENT_SCORE, route_rank
from early_gold_experiment import export_result
from pareto_tail_search import pareto_front


OUTPUT = Path("global_multistart_best_result.json")
START_COUNT = 12
BEAM_WIDTH = 32
MAX_LEVELS = 45
MAX_SOLVES = 160_000
TIME_LIMIT = 600
VALIDATION_SECONDS = 25
MISSION_END = 1827
TANK = 10_000.0
FLIGHT_DAYS = (8, 14, 24, 36, 56, 84)
PER_MATERIAL = 4
EPHEMERIS_STEP = 8


def initial_states(catalog, state_type, stop_type):
    # The evaluator accepts any in-range initial ID, starts with a full tank,
    # and credits mining only from the second entry. Use its day-0 convention.
    order = sorted(range(len(catalog)), key=lambda i: (catalog[i][1], i))
    if len(order) < START_COUNT:
        raise ValueError("The catalog has fewer than 12 distinct starting asteroids.")
    ids = [order[(2 * index + 1) * len(order) // (2 * START_COUNT)]
           for index in range(START_COUNT)]
    if 300 < len(order) and 300 not in ids:
        sample = min(range(START_COUNT), key=lambda n: abs(catalog[ids[n]][1] - catalog[300][1]))
        ids[sample] = 300
    return [
        state_type(i, 1, TANK, (0.0, 0.0, 0.0), frozenset({i}), (stop_type(i, 0, 1),))
        for i in ids
    ]


class SearchLimit(RuntimeError):
    pass


class TransferBudget:
    def __init__(self, calculate, deadline):
        self.calculate = calculate
        self.deadline = deadline
        self.solves = 0
        self.seconds = 0.0
        self.transfer = lru_cache(maxsize=MAX_SOLVES)(self._solve)

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise SearchLimit("search time limit reached")

    def _solve(self, source, departure, target, arrival):
        self.check_time()
        if self.solves >= MAX_SOLVES - 2 * MAX_LEVELS:
            raise SearchLimit("Lambert budget reached")
        self.solves += 1
        started = time.monotonic()
        try:
            return self.calculate(source, target, departure, arrival)
        except (RuntimeError, ValueError):
            return math.nan
        finally:
            self.seconds += time.monotonic() - started


class CatalogShortlist:
    """Use coarse ephemerides only for ranking. Every accepted leg uses exact epochs."""

    def __init__(self, np, udp, epoch, check_time):
        self.np = np
        self.udp = udp
        self.epoch = epoch
        self.check_time = check_time
        self.masses = np.asarray(udp.asteroid_masses)
        materials = np.asarray(udp.asteroid_materials_types)
        self.groups = [np.flatnonzero(materials == kind) for kind in range(4)]

    @lru_cache(maxsize=MISSION_END // EPHEMERIS_STEP + 2)
    def ephemerides(self, day):
        positions, velocities = [], []
        for index, asteroid in enumerate(self.udp.asteroids):
            if index % 256 == 0:
                self.check_time()
            position, velocity = asteroid.eph(self.epoch + day)
            positions.append(position)
            velocities.append(velocity)
        return self.np.asarray(positions), self.np.asarray(velocities)

    def __call__(self, state, arrival, kinds=range(4), per_material=PER_MATERIAL):
        np = self.np
        day = EPHEMERIS_STEP * round(arrival / EPHEMERIS_STEP)
        positions, velocities = self.ephemerides(day)
        source_position, source_velocity = self.udp.asteroids[state.current_id].eph(self.epoch + day)
        flight_seconds = (arrival - state.current_day) * 86400
        excluded = state.visited_asteroids
        ids = []
        for kind in kinds:
            group = self.groups[kind]
            geometry = (
                np.linalg.norm(positions[group] - source_position, axis=1) / flight_seconds
                + np.linalg.norm(velocities[group] - source_velocity, axis=1)
            )
            mass = self.masses[group]
            if kind == 3:
                headroom = TANK - np.maximum(0, state.remaining_dv - geometry)
                utility = np.minimum(mass, headroom / TANK)
            else:
                useful = max(0.15, max(state.collected_materials) - state.collected_materials[kind])
                utility = np.minimum(mass, useful) + 0.25 * mass
            priority = (1 + utility) / (1 + geometry / 2500)
            eligible = (int(group[i]) for i in np.argsort(-priority, kind="stable")
                        if int(group[i]) not in excluded)
            for _ in range(per_material):
                asteroid_id = next(eligible, None)
                if asteroid_id is None:
                    break
                ids.append(asteroid_id)
        return ids


def stays_for(mass, material, arrival, fuel_on_arrival):
    limit = min(60, MISSION_END - arrival)
    if limit < 1:
        return ()
    full = max(1, min(limit, math.ceil(mass * 30)))
    useful = full
    if material == 3:
        useful = max(1, min(full, math.ceil((TANK - fuel_on_arrival) * 30 / TANK)))
    return sorted({max(1, useful // 2), max(1, (3 * useful) // 4), useful})


def projected_minimum(state):
    low, mid, high = sorted(state.collected_materials)
    future_mass = (MISSION_END - state.current_day) * 0.28 / 30
    if future_mass <= mid - low:
        return low + future_mass
    if future_mass <= 2 * high - low - mid:
        return (low + mid + future_mass) / 2
    return (low + mid + high + future_mass) / 3


def select_beam(states, fuel_bonus, stats):
    front = pareto_front(states)
    stats["pareto"] += len(states) - len(front)
    ranked = sorted(front, key=lambda state: (
        projected_minimum(state) + 0.18 * state.remaining_dv / TANK + fuel_bonus.get(state, 0),
        *route_rank(state), -state.current_day,
    ), reverse=True)
    selected = {}
    origins, timings = set(), set()
    # Reserve one representative for each surviving starting region.
    for state in ranked:
        if len(selected) >= BEAM_WIDTH:
            break
        origin = state.route[0].asteroid_id
        if origin not in origins:
            selected[state] = None
            origins.add(origin)
            timings.add((state.current_id, state.current_day // EPHEMERIS_STEP))
    for state in ranked:
        if len(selected) >= BEAM_WIDTH:
            break
        timing = state.current_id, state.current_day // EPHEMERIS_STEP
        if timing not in timings:
            selected[state] = None
            timings.add(timing)
    for state in ranked:
        if len(selected) >= BEAM_WIDTH:
            break
        selected[state] = None
    stats["beam width"] += len(front) - len(selected)
    return list(selected)


class GlobalSearch:
    def __init__(self, engine, shortlist, budget):
        self.engine = engine
        self.shortlist = shortlist
        self.budget = budget
        self.stats = Counter()
        self.best_by_start = {}
        self.shortlist_seconds = 0.0
        self.levels = 0
        self.stop_reason = "level limit"

    def candidates(self, state, arrival, **options):
        started = time.monotonic()
        try:
            return self.shortlist(state, arrival, **options)
        finally:
            self.shortlist_seconds += time.monotonic() - started

    def edge(self, parent, target, arrival, *, lookahead=False):
        dv = self.budget.transfer(parent.current_id, parent.current_day, target, arrival)
        prefix = "lookahead " if lookahead else ""
        if not math.isfinite(dv) or dv < 0:
            self.stats[prefix + "invalid Lambert"] += 1
            return None
        if dv > parent.remaining_dv:
            self.stats[prefix + "insufficient fuel"] += 1
            return None
        return dv

    def expand(self, parent):
        udp = self.engine.udp
        for flight in FLIGHT_DAYS:
            self.budget.check_time()
            arrival = parent.current_day + flight
            if arrival >= MISSION_END:
                self.stats["mission window"] += 1
                continue
            for target in self.candidates(parent, arrival):
                dv = self.edge(parent, target, arrival)
                if dv is None:
                    continue
                material = int(udp.asteroid_materials_types[target])
                mass = float(udp.asteroid_masses[target])
                candidate = {"id": target, "arrival_day": arrival, "dv": dv,
                             "material": material, "mass": mass}
                for stay in stays_for(mass, material, arrival, parent.remaining_dv - dv):
                    child = self.engine.apply_candidate(parent, candidate, stay)
                    if child is None:
                        self.stats["route constraint"] += 1
                    else:
                        yield child, material

    def fuel_lookahead(self, state):
        kind = min(range(3), key=lambda i: state.collected_materials[i])
        for flight in (14, 36):
            self.budget.check_time()
            arrival = state.current_day + flight
            if arrival >= MISSION_END:
                continue
            for target in self.candidates(state, arrival, kinds=(kind,), per_material=1):
                if self.edge(state, target, arrival, lookahead=True) is not None:
                    self.stats["fuel continuation found"] += 1
                    return 0.12
        self.stats["fuel continuation not sampled"] += 1
        return -0.06

    def run(self, roots):
        beam = roots
        try:
            for level in range(1, MAX_LEVELS + 1):
                children = []
                bonuses = {}
                for parent in beam:
                    fuel_children = []
                    expanded = False
                    for child, material in self.expand(parent):
                        expanded = True
                        self.stats["feasible routes"] += 1
                        origin = child.route[0].asteroid_id
                        previous = self.best_by_start.get(origin)
                        if previous is None or route_rank(child) > route_rank(previous):
                            self.best_by_start[origin] = child
                        if child.current_day >= MISSION_END or level == MAX_LEVELS:
                            self.stats["terminal branches"] += 1
                            continue
                        children.append(child)
                        if material == 3:
                            # Compare net refill per elapsed day, including the incoming leg.
                            net = (child.remaining_dv - parent.remaining_dv) / (child.current_day - parent.current_day)
                            fuel_children.append((net, child))
                            bonuses[child] = -0.06
                    if not expanded:
                        self.stats["terminal branches"] += 1
                    for _, state in sorted(fuel_children, key=lambda item: item[0], reverse=True)[:2]:
                        bonuses[state] = self.fuel_lookahead(state)
                self.levels = level
                if not children:
                    self.stop_reason = "level limit" if level == MAX_LEVELS else "frontier exhausted"
                    break
                beam = select_beam(children, bonuses, self.stats)
        except SearchLimit as error:
            self.stop_reason = str(error)
        return sorted(self.best_by_start.values(), key=route_rank, reverse=True)


def run(deadline, started):
    import numpy as np
    import beam_search as engine

    roots = initial_states(engine.esa["data"], engine.BeamState, engine.RouteStop)
    budget = TransferBudget(engine.calculate_transfer_dv, deadline - VALIDATION_SECONDS)
    shortlist = CatalogShortlist(np, engine.udp, engine.esa["T_START"].mjd2000, budget.check_time)
    search = GlobalSearch(engine, shortlist, budget)
    print(f"Start asteroids: {[state.current_id for state in roots]}")
    finalists = search.run(roots)
    print(f"Search: {search.stop_reason} | completed levels {search.levels}"
          f" | Lambert solves {budget.solves} | cache hits {budget.transfer.cache_info().hits}"
          f" | shortlist time {search.shortlist_seconds:.2f} s | Lambert time {budget.seconds:.2f} s")
    print(f"Search stats: {dict(search.stats)}")
    if finalists:
        print(f"Best screened score: {route_rank(finalists[0])[0]:.15f}")
    for state in finalists:
        print(f"Best feasible itinerary from {state.route[0].asteroid_id}:"
              f" score {route_rank(state)[0]:.12f} | day {state.current_day}"
              f" | fuel {state.remaining_dv:.2f} m/s | IDs {[stop.asteroid_id for stop in state.route]}")
    if not finalists or route_rank(finalists[0])[0] <= INCUMBENT_SCORE:
        print("No screened improvement. No official validation used.")
    elif time.monotonic() < deadline:
        validate_and_save(engine, finalists[0], deadline)
    print(f"Elapsed: {time.monotonic() - started:.2f} s")


def validate_and_save(engine, best, deadline):
    try:
        score = engine.validate_route(best, require_full_route=True)
    except (RuntimeError, ValueError) as error:
        print(f"Official validation rejected winner: {error}")
        return
    print(f"Official ESA score: {score!r}")
    if not math.isfinite(score) or score <= INCUMBENT_SCORE or time.monotonic() >= deadline:
        return
    try:
        export_result(best, score, output=OUTPUT, algorithm="Global Multi-Start Beam Search")
    except FileExistsError:
        print(f"{OUTPUT} already exists. Kept the existing result.")
    else:
        print(f"Saved {OUTPUT}")


def main():
    started = time.monotonic()
    deadline = started + TIME_LIMIT
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", "-c",
             "import sys; from global_multistart_beam import run; run(*map(float, sys.argv[1:]))",
             str(deadline), str(started)],
            timeout=max(0.0, deadline - time.monotonic()),
        )
    except subprocess.TimeoutExpired:
        print(f"{TIME_LIMIT}-second limit reached. Worker stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    sys.exit(main())
