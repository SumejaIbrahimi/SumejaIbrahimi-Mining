"""Repair the incumbent's middle refuels and verify the complete continuation."""

from collections import defaultdict, deque
from itertools import zip_longest
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from early_gold_experiment import export_result
from four_stop_incumbent_refinement import KINDS, Refinement, nearby_days
from four_stop_suffix_search import Budget, candidate, check_result, replay, result_rank
from global_multistart_beam import (
    CatalogShortlist, MAX_SOLVES, MISSION_END, SearchLimit, TANK, TIME_LIMIT,
    VALIDATION_SECONDS,
)
from pareto_tail_search import pareto_front
from six_stop_suffix_search import route_arrays


SOURCE = Path(__file__).with_name("four_stop_incumbent_best_result.json")
OUTPUT = Path(__file__).with_name("middle_route_repair_best_result.json")
BASELINE = 8.839573041075445
ANCHOR = (5231, 1328)
REFUELS = (5028, 7002)
REJOIN = 5694
TAIL_ANCHOR = 7230
REJOIN_WINDOW = (1375, 1405)
FUEL_TARGETS = 8
FUEL_WIDTH = 24
REJOIN_WIDTH = 32
CONTINUATION_WIDTH = 32


def reference_indices(states):
    ids = [state.current_id for state in states]
    try:
        start, rejoin, tail = (ids.index(target) for target in (ANCHOR[0], REJOIN, TAIL_ANCHOR))
    except ValueError as error:
        raise ValueError("Missing incumbent waypoint") from error
    if (states[start].current_day != ANCHOR[1]
            or tuple(ids[start + 1:rejoin]) != REFUELS
            or not rejoin < tail == len(states) - 5):
        raise ValueError("Unexpected incumbent repair window")
    return start, rejoin, tail


def check_complete(reference, indices, final, udp):
    start, rejoin, tail = indices
    if final.route[:start + 1] != reference[start].route:
        raise ValueError("Route changed the verified day-1328 prefix")
    suffix = final.route[start + 1:]
    fuel_count = len(suffix) - (len(reference) - rejoin)
    if fuel_count not in (1, 2):
        raise ValueError("Repair must use one or two fuel stops")
    kinds = tuple(int(udp.asteroid_materials_types[stop.asteroid_id]) for stop in suffix)
    if any(kind != 3 for kind in kinds[:fuel_count]):
        raise ValueError("Repair visited a non-fuel asteroid before rejoin")
    fixed = tuple(state.current_id for state in reference[rejoin:tail + 1])
    if tuple(stop.asteroid_id for stop in suffix[fuel_count:-4]) != fixed:
        raise ValueError("Route did not reconnect to the incumbent downstream sequence")
    if kinds[-4:] != KINDS or suffix[-4].asteroid_id != reference[tail + 1].current_id:
        raise ValueError("Unexpected terminal refinement sequence")


def load_reference(saved, engine):
    if saved["official_score"] != BASELINE:
        raise ValueError("Unexpected incumbent score")
    states, minimum = replay(saved, engine)
    indices = reference_indices(states)
    check_complete(states, indices, states[-1], engine.udp)
    score = engine.validate_route(states[-1], require_full_route=True)
    check_result(saved, states[-1], score)
    return states, indices, minimum, score


def select_rejoin(states, width, protected=None):
    """Spread slots across exact days after pruning only identical contexts."""
    by_day = defaultdict(deque)
    ranked = sorted(pareto_front(states), key=lambda state: (
        state.remaining_dv / (TANK / 30) - state.current_day,
        min(state.collected_materials), state.remaining_dv,
    ), reverse=True)
    selected = [] if protected is None else [protected]
    for state in ranked:
        if state != protected:
            by_day[state.current_day].append(state)
    queues = deque(by_day.values())
    while queues and len(selected) < width:
        queue = queues.popleft()
        selected.append(queue.popleft())
        if queue:
            queues.append(queue)
    return selected


def sweep(expansions):
    """Collect completed batches fairly across parents up to the current budget."""
    pending = deque(expansions)
    children = []
    reason = "candidates exhausted"
    try:
        while pending:
            expansion = pending.popleft()
            batch = next(expansion, None)
            if batch is not None:
                children.extend(batch)
                pending.append(expansion)
    except SearchLimit as error:
        reason = str(error)
    return children, reason


class RepairSearch:
    def __init__(self, engine, shortlist, budget, reference, indices):
        self.engine = engine
        self.shortlist = shortlist
        self.budget = budget
        self.reference = reference
        self.start, self.rejoin, self.tail = indices
        self.refinement = Refinement(engine, shortlist, budget, reference[self.tail:])
        self.transfer = self.refinement.transfer
        self.downstream_ids = {state.current_id for state in reference[self.rejoin + 1:self.tail + 1]}

    def option(self, parent, target, arrival):
        self.budget.check_time()
        if target in parent.visited_asteroids or not parent.current_day < arrival <= MISSION_END:
            return None
        dv = self.transfer(parent.current_id, target, parent.current_day, arrival)
        if not 0 <= dv <= parent.remaining_dv:
            return None
        return candidate(self.engine.udp, target, arrival, dv)

    def fuel_children(self, parent, stage):
        center = self.reference[self.start + stage + 1].route[-1].arrival_day
        deadline = REJOIN_WINDOW[1]
        for arrival in nearby_days(center, 24, parent.current_day + 8,
                                   min(parent.current_day + 36, deadline - 9)):
            self.budget.check_time()
            targets = self.shortlist(parent, arrival, kinds=(3,), per_material=FUEL_TARGETS)
            for target in dict.fromkeys([*REFUELS, *targets]):
                if target in self.downstream_ids:
                    continue
                option = self.option(parent, target, arrival)
                if option is None:
                    yield ()
                    continue
                headroom = (TANK - parent.remaining_dv + option["dv"]) / TANK
                full = max(1, math.ceil(30 * min(option["mass"], headroom)))
                latest = min(full + 3, deadline - arrival - 8)
                durations = set(range(1, latest + 1))
                reference_stay = self.reference[self.start + stage + 1].route[-1].mining_days
                if reference_stay <= deadline - arrival - 8:
                    durations.add(reference_stay)
                children = []
                for stay in sorted(durations):
                    child = self.engine.apply_candidate(parent, option, stay)
                    if child is not None:
                        children.append(child)
                yield children

    def fixed_children(self, parent, reference, arrivals):
        stop = reference.route[-1]
        for arrival in arrivals:
            option = self.option(parent, stop.asteroid_id, arrival)
            child = None if option is None else self.engine.apply_candidate(
                parent, option, stop.mining_days,
            )
            yield () if child is None else (child,)

    def rejoin_children(self, parent):
        reference = self.reference[self.rejoin]
        arrival = reference.route[-1].arrival_day
        earliest = max(REJOIN_WINDOW[0], parent.current_day + 8)
        days = nearby_days(arrival, 30, earliest, REJOIN_WINDOW[1])
        return self.fixed_children(parent, reference, days)

    def downstream_children(self, parent, previous, reference):
        arrival = reference.route[-1].arrival_day
        shifted = arrival + parent.current_day - previous.current_day
        earliest, latest = parent.current_day + 1, MISSION_END - reference.route[-1].mining_days
        days = nearby_days(shifted, 3, earliest, latest)
        days.extend(nearby_days(arrival, 2, earliest, latest))
        return self.fixed_children(parent, reference, dict.fromkeys(days))

    def repair(self):
        root = self.reference[self.start]
        first, first_reason = sweep([self.fuel_children(root, 0)])
        first = select_rejoin(first, FUEL_WIDTH, self.reference[self.start + 1])
        second, second_reason = sweep(self.fuel_children(state, 1) for state in first)
        second = select_rejoin(second, FUEL_WIDTH, self.reference[self.start + 2])
        children, reason = sweep(self.rejoin_children(state)
                                 for pair in zip_longest(first, second)
                                 for state in pair if state is not None)
        # Keep a small allocation for each successful refuel count at rejoin.
        by_count = defaultdict(list)
        for state in children:
            by_count[len(state.route) - len(root.route) - 1].append(state)
        selected = [self.reference[self.rejoin]]
        for count in (1, 2):
            selected.extend(state for state in select_rejoin(by_count[count], REJOIN_WIDTH // 4)
                            if state not in selected)
        selected.extend(state for state in select_rejoin(children, REJOIN_WIDTH)
                        if state not in selected)
        print(f"Repair: first refuel {first_reason}; second refuel {second_reason}; "
              f"rejoin {reason} | one-refuel candidates {len(by_count[1])}, "
              f"two-refuel candidates {len(by_count[2])}")
        return selected[:REJOIN_WIDTH]

    def continue_route(self, frontier, solve_end, time_end):
        for index in range(self.rejoin + 1, self.tail + 1):
            set_stage_budget(self.budget, solve_end, time_end, self.tail - index + 1)
            previous, reference = self.reference[index - 1:index + 1]
            children, reason = sweep(self.downstream_children(parent, previous, reference)
                                     for parent in frontier)
            frontier = select_rejoin(children, CONTINUATION_WIDTH, reference)
            print(f"Downstream {reference.current_id}: retained {len(frontier)} | {reason}")
        return frontier

    def finish(self, frontier, solve_end, time_end):
        best = self.reference[-1]
        for stage in range(len(KINDS)):
            left = len(KINDS) - stage
            set_stage_budget(self.budget, solve_end, time_end, left)
            layer_end = self.budget.solve_limit
            if left > 1:
                self.budget.solve_limit -= min(1536, (layer_end - self.budget.solves) // 5)
            children, reason = sweep(self.refinement.expand(state, stage) for state in frontier)
            self.budget.solve_limit = layer_end
            if left > 1:
                frontier = self.refinement.select(children, stage)
            else:
                best = max(children, key=result_rank, default=best)
                if result_rank(best) <= result_rank(self.reference[-1]):
                    best = self.reference[-1]
            print(f"Final stop {stage + 1}: accepted {len(children)} | {reason}")
        return best

    def run(self):
        solve_start, solve_end = self.budget.solves, self.budget.solve_limit
        started, time_end = time.perf_counter(), self.budget.deadline
        repair_end = solve_start + (solve_end - solve_start) * 30 // 100
        downstream_end = solve_start + (solve_end - solve_start) * 70 // 100
        repair_time = started + (time_end - started) * 0.30
        downstream_time = started + (time_end - started) * 0.70
        try:
            self.budget.solve_limit, self.budget.deadline = repair_end, repair_time
            frontier = self.repair()
            frontier = self.continue_route(frontier, downstream_end, downstream_time)
            return self.finish(frontier, solve_end, time_end)
        finally:
            self.budget.solve_limit, self.budget.deadline = solve_end, time_end


def set_stage_budget(budget, solve_end, time_end, stages_left):
    now = time.perf_counter()
    budget.solve_limit = budget.solves + max(0, solve_end - budget.solves) // stages_left
    budget.deadline = min(time_end, now + max(0, time_end - now) / stages_left)


def report(state, score, minimum, budget, started):
    print(f"Official score: {score!r} | improvement: {score - BASELINE:+.15f}")
    print(f"Materials: {state.collected_materials} | final day: {state.current_day}")
    print(f"Remaining fuel: {state.remaining_dv:.6f} m/s | minimum after transfer: {minimum:.6f} m/s")
    print(f"Lambert solves: {budget.solves} | runtime: {time.perf_counter() - started:.2f} s")


def run(started, deadline):
    import beam_search as engine

    budget = Budget(engine.pk.lambert_problem, deadline)
    engine.pk.lambert_problem = budget
    try:
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        reference, indices, incumbent_minimum, incumbent_score = load_reference(saved, engine)
        anchor = reference[indices[0]]
        print(f"Verified anchor: asteroid {anchor.current_id}, day {anchor.current_day}, "
              f"fuel {anchor.remaining_dv:.6f} m/s")
        budget.solve_limit = MAX_SOLVES - 5 * (len(reference) - 1)
        budget.deadline = deadline - VALIDATION_SECONDS
        shortlist = CatalogShortlist(engine.np, engine.udp, engine.esa["T_START"].mjd2000,
                                     budget.check_time)
        best = RepairSearch(engine, shortlist, budget, reference, indices).run()
        budget.solve_limit, budget.deadline = MAX_SOLVES, deadline
        if min(best.collected_materials) <= BASELINE:
            print("No screened improvement. Reporting the verified incumbent.")
            report(reference[-1], incumbent_score, incumbent_minimum, budget, started)
            return 0

        check_complete(reference, indices, best, engine.udp)
        official = engine.validate_route(best, require_full_route=True)
        states, minimum = replay(json.loads(json.dumps(route_arrays(best), allow_nan=False)), engine)
        final = states[-1]
        check_complete(reference, indices, final, engine.udp)
        replayed = engine.validate_route(final, require_full_route=True)
        check_result({
            "score": min(best.collected_materials), "official_score": official,
            "remaining_dv": best.remaining_dv, "final_day": best.current_day,
            "collected_materials": {str(i): mass for i, mass in enumerate(best.collected_materials)},
        }, final, replayed)
        if min(official, replayed) <= BASELINE:
            print("Official validation did not confirm an improvement.")
            report(reference[-1], incumbent_score, incumbent_minimum, budget, started)
            return 0
        budget.check_time()
        try:
            export_result(final, replayed, output=OUTPUT, algorithm="Incumbent-seeded middle-route repair")
        except FileExistsError:
            print(f"{OUTPUT.name} already exists and was preserved.")
        else:
            print(f"Saved {OUTPUT.name}")
        report(final, replayed, minimum, budget, started)
        return 0
    except (RuntimeError, ValueError) as error:
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
