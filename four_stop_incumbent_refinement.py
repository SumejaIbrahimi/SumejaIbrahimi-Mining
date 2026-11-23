"""Refine the verified Gold, Fuel, Platinum, Nickel finish around its saved schedule."""

from collections import defaultdict, deque
from functools import lru_cache
from itertools import islice
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from early_gold_experiment import export_result
from four_stop_suffix_search import Budget, candidate, check_result, replay, result_rank
from global_multistart_beam import (
    CatalogShortlist, MAX_SOLVES, MISSION_END, SearchLimit, TANK, TIME_LIMIT,
    VALIDATION_SECONDS,
)
from pareto_tail_search import pareto_front
from six_stop_suffix_search import mining_days, route_arrays


SOURCE = Path(__file__).with_name("six_stop_suffix_best_result.json")
OUTPUT = Path(__file__).with_name("four_stop_incumbent_best_result.json")
BASELINE = 8.772906374408777
ANCHOR = (7230, 1670)
INCUMBENT_IDS = (6170, 5137, 2787, 3094)
KINDS = (0, 3, 1, 2)
RADII = (7, 12, 14, 12)
BEAM_WIDTH = 64
TARGETS = 10
GOAL = 9.0
EPSILON = 1e-10


def reconstruct_anchor(saved, engine):
    if saved["official_score"] != BASELINE:
        raise ValueError("Unexpected incumbent score")
    states, minimum = replay(saved, engine)
    tail = states[-5:]
    if (len(tail) != 5 or (tail[0].current_id, tail[0].current_day) != ANCHOR
            or tuple(state.current_id for state in tail[1:]) != INCUMBENT_IDS):
        raise ValueError("Unexpected day-1670 anchor or incumbent suffix")
    check_prefix(tail[0], tail[-1], engine.udp)
    official = engine.validate_route(tail[-1], require_full_route=True)
    check_result(saved, tail[-1], official)
    return tail, minimum, official


def check_prefix(anchor, final, udp):
    prefix_length = len(anchor.route)
    if final.route[:prefix_length] != anchor.route:
        raise ValueError("Route changed the verified prefix")
    suffix = final.route[prefix_length:]
    kinds = tuple(int(udp.asteroid_materials_types[stop.asteroid_id]) for stop in suffix)
    if kinds != KINDS or suffix[0].asteroid_id != INCUMBENT_IDS[0]:
        raise ValueError("Unexpected refinement stop sequence")


def nearby_days(center, radius, earliest, latest):
    return sorted(range(max(earliest, center - radius), min(latest, center + radius) + 1),
                  key=lambda day: (abs(day - center), day))


class Refinement:
    def __init__(self, engine, shortlist, budget, reference):
        self.engine = engine
        self.shortlist = shortlist
        self.budget = budget
        self.reference = reference
        self.anchor = reference[0]
        self.transfer = lru_cache(maxsize=MAX_SOLVES)(self.calculate_transfer)
        self.high_yield_groups = {}
        for kind in (1, 2):
            total = self.anchor.collected_materials[kind]
            group = [int(target) for target in shortlist.groups[kind]
                     if int(target) not in self.anchor.visited_asteroids
                     and total + shortlist.masses[target] >= BASELINE - EPSILON]
            shortlist.groups[kind] = group
            self.high_yield_groups[kind] = [
                target for target in group if total + shortlist.masses[target] > GOAL
            ]

    def calculate_transfer(self, source, target, departure, arrival):
        try:
            return self.engine.calculate_transfer_dv(source, target, departure, arrival)
        except SearchLimit:
            raise
        except (ValueError, RuntimeError):
            return math.nan

    def targets(self, state, arrival, kind, incumbent):
        if kind == 0:
            return (incumbent,)
        ids = self.shortlist(state, arrival, kinds=(kind,), per_material=TARGETS)
        high_yield = self.high_yield_groups.get(kind)
        if high_yield:
            # Give high-deposit targets dedicated slots in addition to the general shortlist.
            original = self.shortlist.groups[kind]
            self.shortlist.groups[kind] = high_yield
            try:
                high_ids = self.shortlist(state, arrival, kinds=(kind,), per_material=TARGETS // 2)
            finally:
                self.shortlist.groups[kind] = original
            ids.extend(high_ids)
        return dict.fromkeys([incumbent, *ids])

    def edges(self, state, stage, per_arrival=None):
        kind = KINDS[stage]
        stop = self.reference[stage + 1].route[-1]
        later = KINDS[stage + 1:]
        reserve = sum(
            1 + (1 if material == 3 else max(1, mining_days(BASELINE - state.collected_materials[material])))
            for material in later
        )
        minimum = 1 if kind == 3 else max(1, mining_days(BASELINE - state.collected_materials[kind]))
        for arrival in nearby_days(stop.arrival_day, RADII[stage], state.current_day + 1,
                                   MISSION_END - reserve - minimum):
            self.budget.check_time()
            maximum = min(60, MISSION_END - arrival - reserve)
            for target in islice(self.targets(state, arrival, kind, stop.asteroid_id), per_arrival):
                if target in state.visited_asteroids:
                    continue
                option = candidate(self.engine.udp, target, arrival, 0)
                if kind != 3 and state.collected_materials[kind] + option["mass"] < BASELINE - EPSILON:
                    continue
                option["dv"] = self.transfer(state.current_id, target, state.current_day, arrival)
                yield option, minimum, maximum

    def stays(self, state, stage, option, minimum, maximum):
        kind = KINDS[stage]
        if stage == len(KINDS) - 1:
            return (maximum,)
        useful = option["mass"]
        if kind == 3:
            arrival_fuel = state.remaining_dv - option["dv"]
            useful = min(useful, (TANK - arrival_fuel) / TANK)
        full = min(maximum, max(minimum, math.ceil(30 * useful)))
        # Keep nearby whole-day departures even a little beyond saturation.
        reference_stay = self.reference[stage + 1].route[-1].mining_days
        stays = set(range(minimum, min(maximum, full + 3) + 1))
        if minimum <= reference_stay <= maximum:
            stays.add(reference_stay)
        return sorted(stays)

    def expand(self, state, stage):
        kind = KINDS[stage]
        for option, minimum, maximum in self.edges(state, stage):
            children = []
            if 0 <= option["dv"] <= state.remaining_dv:
                for stay in self.stays(state, stage, option, minimum, maximum):
                    child = self.engine.apply_candidate(state, option, stay)
                    if child is not None and (kind == 3 or child.collected_materials[kind] >= BASELINE - EPSILON):
                        children.append(child)
            yield children

    def priority(self, state, stage):
        completed = [state.collected_materials[kind] for kind in KINDS[:stage + 1] if kind != 3]
        ceiling = min(completed, default=GOAL)
        timing_fuel = state.remaining_dv / (TANK / 30) - state.current_day
        return min(ceiling, GOAL + 0.05), timing_fuel

    def lookahead(self, state, stage):
        best = -1.0
        for option, _, maximum in islice(self.edges(state, stage, per_arrival=3), 12):
            fuel = state.remaining_dv - option["dv"]
            if not 0 <= fuel <= TANK:
                continue
            if KINDS[stage] == 3:
                fuel = min(TANK, fuel + TANK * min(option["mass"], maximum / 30))
            best = max(best, fuel / TANK)
        return best

    def select(self, children, stage):
        protected = self.reference[stage + 1]
        groups = defaultdict(deque)
        ranked = sorted(pareto_front(children), key=lambda state: self.priority(state, stage), reverse=True)
        for state in ranked:
            if state != protected:
                groups[state.current_id].append(state)
        # Round-robin by endpoint before lookahead. Distinct exact days survive
        # Pareto pruning and can compete for the available beam slots.
        queues = deque(groups.values())
        pool = []
        while queues and len(pool) < 2 * BEAM_WIDTH:
            queue = queues.popleft()
            pool.append(queue.popleft())
            if queue:
                queues.append(queue)
        ranked = []
        for state in pool:
            try:
                continuation = self.lookahead(state, stage + 1)
            except SearchLimit:
                continuation = -1.0
            ranked.append((continuation, state))
        ranked.sort(key=lambda item: (item[0] >= 0, self.priority(item[1], stage), item[0]),
                    reverse=True)
        return [protected, *(state for _, state in ranked[:BEAM_WIDTH - 1])]

    def run(self):
        frontier = [self.anchor]
        best = self.reference[-1]
        solve_limit, deadline = self.budget.solve_limit, self.budget.deadline
        try:
            for stage in range(len(KINDS)):
                before = self.budget.solves
                layers_left = len(KINDS) - stage
                allowance = (solve_limit - before) // layers_left
                layer_end = before + allowance
                lookahead_reserve = min(1536, allowance // 5) if layers_left > 1 else 0
                self.budget.solve_limit = layer_end - lookahead_reserve
                now = time.perf_counter()
                self.budget.deadline = min(deadline, now + max(0, deadline - now) / layers_left)
                pending = deque(self.expand(state, stage) for state in frontier)
                children = []
                completed = 0
                reason = "candidates exhausted"
                try:
                    while pending:
                        sweep = pending.popleft()
                        batch = next(sweep, None)
                        if batch is None:
                            continue
                        pending.append(sweep)
                        if layers_left > 1:
                            children.extend(batch)
                        else:
                            completed += len(batch)
                            for state in batch:
                                if result_rank(state) > result_rank(best):
                                    best = state
                except SearchLimit as error:
                    reason = str(error)
                self.budget.solve_limit = layer_end
                frontier = self.select(children, stage) if layers_left > 1 else []
                print(f"Layer {stage + 1}: Lambert solves {self.budget.solves - before}, "
                      f"children {len(children)}, retained {len(frontier)}, "
                      f"complete {completed} | {reason}")
                if time.perf_counter() >= deadline or self.budget.solves >= solve_limit:
                    break
        finally:
            self.budget.solve_limit, self.budget.deadline = solve_limit, deadline
        return best


def report(state, score, minimum, budget, started):
    print(f"Official score: {score!r} | improvement: {score - BASELINE:+.15f}")
    print(f"Material totals: {state.collected_materials} | final day: {state.current_day}")
    print(f"Remaining fuel: {state.remaining_dv:.6f} m/s | minimum after transfer: {minimum:.6f} m/s")
    print(f"Lambert solves: {budget.solves} | runtime: {time.perf_counter() - started:.2f} s")


def run(started, deadline):
    import beam_search as engine

    budget = Budget(engine.pk.lambert_problem, deadline)
    engine.pk.lambert_problem = budget
    try:
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        reference, minimum, incumbent_score = reconstruct_anchor(saved, engine)
        anchor, incumbent = reference[0], reference[-1]
        print(f"Verified anchor: asteroid {anchor.current_id}, day {anchor.current_day}, "
              f"fuel {anchor.remaining_dv:.6f} m/s, materials {anchor.collected_materials}")
        budget.solve_limit = MAX_SOLVES - 5 * (len(incumbent.route) - 1)
        budget.deadline = deadline - VALIDATION_SECONDS
        shortlist = CatalogShortlist(engine.np, engine.udp, engine.esa["T_START"].mjd2000,
                                     budget.check_time)
        best = Refinement(engine, shortlist, budget, reference).run()
        budget.solve_limit, budget.deadline = MAX_SOLVES, deadline
        if min(best.collected_materials) <= BASELINE:
            print("No screened improvement. Reporting the verified incumbent.")
            report(incumbent, incumbent_score, minimum, budget, started)
            return 0

        check_prefix(anchor, best, engine.udp)
        official = engine.validate_route(best, require_full_route=True)
        states, minimum = replay(json.loads(json.dumps(route_arrays(best), allow_nan=False)), engine)
        final = states[-1]
        check_prefix(anchor, final, engine.udp)
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
            export_result(final, replayed_score, output=OUTPUT, algorithm="Four-stop incumbent refinement")
        except FileExistsError:
            print(f"{OUTPUT.name} already exists and was preserved.")
        else:
            print(f"Saved {OUTPUT.name}")
        report(final, replayed_score, minimum, budget, started)
        return 0
    except (ValueError, RuntimeError) as error:
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
