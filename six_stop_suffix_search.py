"""Search the final six stops after the verified day-1596 departure."""

from collections import defaultdict, deque
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from early_gold_experiment import export_result
from four_stop_suffix_search import (
    BASELINE, SOURCE, Budget, candidate, check_result, replay, report, result_rank,
)
from global_multistart_beam import (
    CatalogShortlist, MAX_SOLVES, MISSION_END, SearchLimit, TANK, TIME_LIMIT,
    VALIDATION_SECONDS,
)
from pareto_tail_search import pareto_front


OUTPUT = Path(__file__).with_name("six_stop_suffix_best_result.json")
ANCHOR = (4274, 1596)
# Catalog order: gold, platinum, nickel, fuel.
PATTERN = (1, 2, 1, 2)
BEAM_WIDTH = 48
TARGETS = 8
FLIGHTS = (8, 12, 16, 20, 24, 30, 36, 48, 60)
LOOKAHEAD_FLIGHTS = (14, 24, 36)
EPSILON = 1e-10


def route_arrays(state):
    return {
        "visited_asteroids": [stop.asteroid_id for stop in state.route],
        "arrival_times": [stop.arrival_day for stop in state.route],
        "mining_times": [stop.mining_days for stop in state.route],
    }


def check_suffix(anchor, final, udp):
    prefix_length = len(anchor.route)
    if final.route[:prefix_length] != anchor.route:
        raise ValueError("Route changed the verified prefix")
    kinds = [int(udp.asteroid_materials_types[stop.asteroid_id])
             for stop in final.route[prefix_length:]]
    if tuple(kinds.count(kind) for kind in range(4)) != PATTERN:
        raise ValueError("Expected one gold, two platinum, one nickel and two fuel stops")


def reconstruct_anchor(saved, engine):
    if saved["official_score"] != BASELINE:
        raise ValueError("Unexpected baseline score")
    states, minimum = replay(saved, engine)
    if len(states) <= sum(PATTERN):
        raise ValueError("Baseline has too few stops")
    anchor = states[-1 - sum(PATTERN)]
    if (anchor.current_id, anchor.current_day) != ANCHOR:
        raise ValueError("Expected asteroid 4274 departing on day 1596")
    incumbent = states[-1]
    check_suffix(anchor, incumbent, engine.udp)
    score = engine.validate_route(incumbent, require_full_route=True)
    check_result(saved, incumbent, score)
    return states[len(anchor.route) - 1:], minimum, score


def mining_days(mass):
    # A downward tolerance keeps a lower bound conservative at integral days.
    return max(0, math.ceil(30 * (mass - EPSILON)))


class SuffixSearch:
    def __init__(self, engine, shortlist, budget, reference):
        self.engine = engine
        self.shortlist = shortlist
        self.budget = budget
        self.reference = reference
        root, incumbent = reference[0], reference[-1]
        self.reference_stops = defaultdict(list)
        for stop in incumbent.route[len(root.route):]:
            kind = int(engine.udp.asteroid_materials_types[stop.asteroid_id])
            self.reference_stops[kind].append(stop)
        self.deposits = [
            sorted(((float(shortlist.masses[target]), int(target)) for target in group
                    if int(target) not in root.visited_asteroids), reverse=True)
            for group in shortlist.groups
        ]
        # Filter before ranking so undersized deposits cannot fill the shortlist.
        # For platinum, credit the best distinct partner optimistically.
        for kind, collected in enumerate(root.collected_materials):
            deposits = self.deposits[kind]
            group = []
            for mass, target in deposits:
                partner = 0.0
                if PATTERN[kind] == 2:
                    partner = next((other for other, asteroid in deposits[:2]
                                    if asteroid != target), 0.0)
                if collected + mass + partner >= BASELINE - EPSILON:
                    group.append(target)
            shortlist.groups[kind] = group
        self.transfer = lru_cache(maxsize=MAX_SOLVES)(self.calculate_transfer)

    def calculate_transfer(self, source, target, departure, arrival):
        try:
            return self.engine.calculate_transfer_dv(source, target, departure, arrival)
        except SearchLimit:
            raise
        except (ValueError, RuntimeError):
            return math.nan

    def largest_deposits(self, kind, count, visited):
        result = []
        for mass, target in self.deposits[kind]:
            if len(result) == count:
                break
            if target not in visited:
                result.append((mass, target))
        return result

    def bounds(self, state, remaining):
        """Optimistic score and days, ignoring geometry and fuel transfer costs."""
        days = sum(remaining) + remaining[3]
        ceilings = []
        for kind, collected in enumerate(state.collected_materials):
            count = remaining[kind]
            deposits = self.largest_deposits(kind, count, state.visited_asteroids)
            if len(deposits) != count:
                return None
            ceilings.append(collected + sum(mass for mass, _ in deposits))
            days += max(count, mining_days(BASELINE - collected))
        ceiling = min(ceilings)
        if ceiling < BASELINE - EPSILON or state.current_day + days > MISSION_END:
            return None
        return ceiling, days

    def edges(self, state, remaining, flights, targets, *, reference=True):
        for kind, count in enumerate(remaining):
            if not count:
                continue
            later = list(remaining)
            later[kind] -= 1
            later = tuple(later)
            reference_stops = self.reference_stops[kind] if reference else ()
            arrivals = {state.current_day + flight for flight in flights}
            arrivals.update(stop.arrival_day for stop in reference_stops)
            # Every arrival leaves at least one mining day and one day per later activity.
            latest_arrival = MISSION_END - 1 - 2 * sum(later)
            for arrival in sorted(arrivals):
                if not state.current_day < arrival <= latest_arrival:
                    continue
                self.budget.check_time()
                ids = self.shortlist(state, arrival, kinds=(kind,), per_material=targets)
                ids.extend(stop.asteroid_id for stop in reference_stops
                           if stop.asteroid_id not in state.visited_asteroids)
                for target in dict.fromkeys(ids):
                    option = candidate(self.engine.udp, target, arrival, 0)
                    minimum = 1
                    if kind != 3:
                        others = self.largest_deposits(
                            kind, later[kind], state.visited_asteroids | {target},
                        )
                        deficit = BASELINE - state.collected_materials[kind]
                        needed = deficit - sum(mass for mass, _ in others)
                        if len(others) != later[kind] or option["mass"] < needed - EPSILON:
                            continue
                        minimum = max(1, mining_days(needed))
                    reserve = sum(later) + later[3]
                    for material, collected in enumerate(state.collected_materials):
                        optimistic = collected
                        if material == kind:
                            optimistic += option["mass"]
                        reserve += max(later[material], mining_days(BASELINE - optimistic))
                    maximum = min(60, MISSION_END - arrival - reserve)
                    if minimum > maximum:
                        continue
                    dv = self.transfer(state.current_id, target, state.current_day, arrival)
                    option["dv"] = dv
                    yield kind, later, option, minimum, maximum

    def expand(self, state, remaining):
        for kind, later, option, minimum, maximum in self.edges(state, remaining, FLIGHTS, TARGETS):
            children = []
            if 0 <= option["dv"] <= state.remaining_dv:
                useful = option["mass"]
                if kind == 3:
                    headroom = TANK - (state.remaining_dv - option["dv"])
                    useful = min(useful, headroom / TANK)
                full = min(maximum, max(minimum, math.ceil(30 * useful)))
                if sum(later):
                    stays = {minimum, (minimum + full) // 2, full, maximum}
                    stays.update(stop.mining_days for stop in self.reference_stops[kind]
                                 if minimum <= stop.mining_days <= maximum)
                else:
                    stays = {maximum}
                for stay in sorted(stays):
                    child = self.engine.apply_candidate(state, option, stay)
                    if child is not None and self.bounds(child, later) is not None:
                        children.append((child, later))
            yield children

    def reachability(self, state, remaining):
        # Cheap exact lookahead. Use arrival fuel and possible refill to score the
        # reachable edge. A missed sample remains eligible for the beam.
        best = -1.0
        probes = self.edges(state, remaining, LOOKAHEAD_FLIGHTS, 1, reference=False)
        for kind, _, option, _, maximum in probes:
            fuel = state.remaining_dv - option["dv"]
            if not 0 <= fuel <= TANK:
                continue
            if kind == 3:
                fuel = min(TANK, fuel + TANK * min(option["mass"], maximum / 30))
            best = max(best, fuel / TANK)
        return best

    def select(self, children):
        groups = defaultdict(list)
        for state, remaining in children:
            groups[remaining].append(state)
        queues = []
        for remaining, states in groups.items():
            ranked = sorted(pareto_front(states), key=lambda state: self.priority(state, remaining),
                            reverse=True)
            queues.append(deque((state, remaining) for state in ranked))
        pool = []
        while queues and len(pool) < 2 * BEAM_WIDTH:
            for queue in queues:
                pool.append(queue.popleft())
                if len(pool) == 2 * BEAM_WIDTH:
                    break
            queues = [queue for queue in queues if queue]
        by_group = defaultdict(list)
        for state, remaining in pool:
            try:
                continuation = self.reachability(state, remaining)
            except SearchLimit:
                continuation = -1.0
            by_group[remaining].append((continuation, state))
        queues = [
            deque((state, remaining) for _, state in sorted(
                states, key=lambda pair: (pair[0], self.priority(pair[1], remaining)), reverse=True,
            ))
            for remaining, states in by_group.items()
        ]
        selected = []
        while queues and len(selected) < BEAM_WIDTH:
            for queue in queues:
                selected.append(queue.popleft())
                if len(selected) == BEAM_WIDTH:
                    break
            queues = [queue for queue in queues if queue]
        return selected

    def priority(self, state, remaining):
        ceiling, days = self.bounds(state, remaining)
        slack = MISSION_END - state.current_day - days
        return min(ceiling, 9.2), slack + state.remaining_dv / (TANK / 30)

    def run(self):
        frontier = [(self.reference[0], PATTERN)]
        best = self.reference[-1]
        solve_limit, deadline = self.budget.solve_limit, self.budget.deadline
        reference_remaining = list(PATTERN)
        try:
            for layer in range(1, sum(PATTERN) + 1):
                before = self.budget.solves
                now = time.perf_counter()
                layers_left = sum(PATTERN) - layer + 1
                allowance = (solve_limit - before) // layers_left
                layer_end = before + allowance
                self.budget.solve_limit = layer_end - min(1200, allowance // 5) if layers_left > 1 else layer_end
                self.budget.deadline = min(deadline, now + max(0, deadline - now) / layers_left)
                pending = deque(self.expand(state, remaining) for state, remaining in frontier)
                children = []
                completed = 0
                reason = "candidates exhausted"
                try:
                    while pending:
                        expansion = pending.popleft()
                        batch = next(expansion, None)
                        if batch is None:
                            continue
                        pending.append(expansion)
                        for state, remaining in batch:
                            if any(remaining):
                                children.append((state, remaining))
                            else:
                                completed += 1
                                if result_rank(state) > result_rank(best):
                                    best = state
                except SearchLimit as error:
                    reason = str(error)
                self.budget.solve_limit = layer_end
                frontier = self.select(children)
                # Give each verified incumbent prefix a chance to branch at this depth.
                reference = self.reference[layer]
                kind = int(self.engine.udp.asteroid_materials_types[reference.current_id])
                reference_remaining[kind] -= 1
                if layer < sum(PATTERN):
                    protected = (reference, tuple(reference_remaining))
                    if protected not in frontier:
                        frontier = [protected, *frontier[:BEAM_WIDTH - 1]]
                print(f"Layer {layer}: Lambert solves {self.budget.solves - before}, "
                      f"children {len(children)}, retained {len(frontier)}, "
                      f"completed {completed} | {reason}")
                if time.perf_counter() >= deadline or self.budget.solves >= solve_limit:
                    break
        finally:
            self.budget.solve_limit, self.budget.deadline = solve_limit, deadline
        return best


def run(started, deadline):
    import beam_search as engine

    budget = Budget(engine.pk.lambert_problem, deadline)
    engine.pk.lambert_problem = budget
    try:
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        reference, minimum, official = reconstruct_anchor(saved, engine)
        anchor, incumbent = reference[0], reference[-1]
        print(f"Verified anchor: asteroid {anchor.current_id}, day {anchor.current_day}, "
              f"fuel {anchor.remaining_dv:.6f} m/s, materials {anchor.collected_materials}")
        budget.solve_limit = MAX_SOLVES - 5 * (len(incumbent.route) - 1)
        budget.deadline = deadline - VALIDATION_SECONDS
        shortlist = CatalogShortlist(engine.np, engine.udp, engine.esa["T_START"].mjd2000,
                                     budget.check_time)
        best = SuffixSearch(engine, shortlist, budget, reference).run()
        budget.solve_limit, budget.deadline = MAX_SOLVES, deadline
        if min(best.collected_materials) <= BASELINE:
            print("No screened improvement. Reporting the verified incumbent.")
            report(incumbent, official, minimum, budget, started)
            return 0

        check_suffix(anchor, best, engine.udp)
        official = engine.validate_route(best, require_full_route=True)
        arrays = json.loads(json.dumps(route_arrays(best), allow_nan=False))
        states, minimum = replay(arrays, engine)
        final = states[-1]
        check_suffix(anchor, final, engine.udp)
        replayed_score = engine.validate_route(final, require_full_route=True)
        expected = {
            "score": min(best.collected_materials), "official_score": official,
            "remaining_dv": best.remaining_dv, "final_day": best.current_day,
            "collected_materials": {str(i): mass for i, mass in enumerate(best.collected_materials)},
        }
        check_result(expected, final, replayed_score)
        if min(official, replayed_score) <= BASELINE:
            print("Official validation did not confirm an improvement.")
            return 0
        budget.check_time()
        try:
            export_result(final, replayed_score, output=OUTPUT, algorithm="Six-stop suffix search")
        except FileExistsError:
            print(f"{OUTPUT.name} already exists and was preserved.")
        else:
            print(f"Saved {OUTPUT.name}")
        report(final, replayed_score, minimum, budget, started)
        return 0
    except (SearchLimit, ValueError, RuntimeError) as error:
        print(f"{error} | Lambert solves: {budget.solves} "
              f"| runtime: {time.perf_counter() - started:.2f} s")
        return 1
    finally:
        engine.pk.lambert_problem = budget.lambert


def main():
    started = time.perf_counter()
    deadline = started + TIME_LIMIT
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", str(started)],
            cwd=SOURCE.parent, timeout=max(0, deadline - time.perf_counter()),
        )
    except subprocess.TimeoutExpired:
        print(f"{TIME_LIMIT}-second wall-clock limit reached. Worker stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        start = float(sys.argv[2])
        sys.exit(run(start, start + TIME_LIMIT))
    sys.exit(main())
