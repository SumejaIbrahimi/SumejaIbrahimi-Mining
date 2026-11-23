"""Test exact Platinum–Nickel completions from repaired downstream itineraries."""

from collections import Counter, defaultdict, deque
from itertools import zip_longest
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from early_gold_experiment import export_result
from four_stop_incumbent_refinement import KINDS
from four_stop_suffix_search import Budget, candidate, check_result, replay, result_rank
from global_multistart_beam import (
    CatalogShortlist, MAX_SOLVES, MISSION_END, SearchLimit, TANK, TIME_LIMIT,
    VALIDATION_SECONDS,
)
from middle_route_repair import (
    BASELINE, REJOIN, SOURCE, RepairSearch, check_complete, load_reference, report,
    set_stage_budget, sweep,
)
from pareto_tail_search import pareto_front
from six_stop_suffix_search import mining_days, route_arrays


OUTPUT = Path(__file__).with_name("completion_aware_downstream_best_result.json")
GOAL = 9.0
EARLY_WIDTH = 64
PLATINUM_POOL = 512
NICKEL_PROBES = 80
EPSILON = 1e-10


def sampled_days(earliest, latest, preferred, goal_latest, count):
    """Mix local flight timings with the deadline and an evenly spaced date grid."""
    if earliest > latest:
        return ()
    preferred = min(latest, max(earliest, preferred))
    goal_latest = min(latest, max(earliest, goal_latest))
    span = latest - earliest
    grid = [earliest + span * index // 7 for index in (1, 5, 3, 0, 7, 2, 4, 6)]
    days = [preferred, goal_latest, preferred - 1, preferred + 1, goal_latest - 1,
            *grid, preferred - 2, preferred + 2]
    return tuple(dict.fromkeys(day for day in days if earliest <= day <= latest))[:count]


class CompletionSearch(RepairSearch):
    def repair_stops(self, state):
        stops = []
        for stop in state.route[self.start + 1:]:
            if stop.asteroid_id == REJOIN:
                break
            stops.append(stop)
        return tuple(stops)

    def describe_ancestry(self, state):
        rejoin = next(stop for stop in state.route[self.start + 1:] if stop.asteroid_id == REJOIN)
        return ([(stop.asteroid_id, stop.arrival_day, stop.mining_days)
                 for stop in self.repair_stops(state)], rejoin.arrival_day)

    def diverse(self, states, width, rank, protected):
        """Round-robin by refuel lineage and exact departure day."""
        states = sorted(pareto_front(states), key=rank, reverse=True)
        groups = defaultdict(lambda: defaultdict(deque))
        for state in states:
            if state != protected:
                lineage = tuple(stop.asteroid_id for stop in self.repair_stops(state))
                groups[lineage][state.current_day].append(state)
        lineages = deque(deque(days.values()) for days in groups.values())
        selected = [protected]
        while lineages and len(selected) < width:
            days = lineages.popleft()
            fuels = days.popleft()
            selected.append(fuels.popleft())
            if fuels:
                days.append(fuels)
            if days:
                lineages.append(days)
        return selected

    def schedule(self, state, stage):
        kind = KINDS[stage]
        later = KINDS[stage + 1:]
        reserves = []
        for target in (BASELINE, GOAL):
            days = 0
            for material in (kind, *later):
                days += (1 if material == 3 else
                         max(1, mining_days(target - state.collected_materials[material])))
            reserves.append(days + len(later))
        reference = self.reference[self.tail + stage + 1].route[-1]
        previous = self.reference[self.tail + stage]
        preferred = state.current_day + reference.arrival_day - previous.current_day
        dates = sampled_days(state.current_day + 1, MISSION_END - reserves[0],
                             preferred, MISSION_END - reserves[1], 14)
        minimum = 1 if kind == 3 else max(1, mining_days(BASELINE - state.collected_materials[kind]))
        return dates, minimum, reserves[0] - minimum

    def children(self, state, stage, stats):
        dates, minimum, reserve = self.schedule(state, stage)
        if not dates:
            stats["timing window empty"] += 1
        # Give distinct dates probes before spending the per-parent cap on more targets.
        pending = deque(self.arrival_children(state, stage, arrival, minimum, reserve, stats)
                        for arrival in dates)
        while pending:
            expansion = pending.popleft()
            batch = next(expansion, None)
            if batch is not None:
                pending.append(expansion)
                yield batch

    def arrival_children(self, state, stage, arrival, minimum, reserve, stats):
        kind = KINDS[stage]
        reference = self.reference[self.tail + stage + 1].route[-1]
        self.budget.check_time()
        maximum = min(60, MISSION_END - arrival - reserve)
        targets = list(self.refinement.targets(state, arrival, kind, reference.asteroid_id))
        if kind in (1, 2):
            total = state.collected_materials[kind]
            masses = self.engine.udp.asteroid_masses
            rich, general = [], []
            for target in targets[1:]:
                (rich if total + masses[target] > GOAL else general).append(target)
            targets = [targets[0], *(target for pair in zip_longest(rich, general)
                                     for target in pair if target is not None)]
        for target in targets:
            self.budget.check_time()
            stats["options"] += 1
            if target in state.visited_asteroids:
                stats["visited"] += 1
                continue
            mass = float(self.engine.udp.asteroid_masses[target])
            if kind != 3 and state.collected_materials[kind] + mass < BASELINE - EPSILON:
                stats["deposit too small"] += 1
                continue
            if kind != 3 and maximum < minimum:
                stats["insufficient mining time"] += 1
                continue
            stats["transfer attempts"] += 1
            dv = self.transfer(state.current_id, target, state.current_day, arrival)
            if not math.isfinite(dv) or dv < 0:
                stats["invalid Lambert"] += 1
                yield ()
                continue
            if dv > state.remaining_dv:
                stats["insufficient fuel"] += 1
                yield ()
                continue
            option = candidate(self.engine.udp, target, arrival, dv)
            options = []
            # The maximal legal Nickel stay maximizes the complete minimum at this epoch.
            stays = self.refinement.stays(state, stage, option, minimum, maximum)
            for stay in stays:
                child = self.engine.apply_candidate(state, option, stay)
                if child is None:
                    stats["transition"] += 1
                elif kind != 3 and child.collected_materials[kind] < BASELINE - EPSILON:
                    stats["mining yield"] += 1
                else:
                    options.append(child)
            yield options

    def finish(self, frontier, solve_end, time_end):
        incumbent = self.reference[-1]
        original = self.describe_ancestry(incumbent)
        repaired = 0
        for state in frontier:
            ancestry = self.describe_ancestry(state)
            repaired += ancestry != original
            print(f"7230 state: day {state.current_day}, fuel {state.remaining_dv:.3f}, "
                  f"refuels {ancestry[0]}, rejoin day {ancestry[1]}")
        print(f"7230 retained: {len(frontier)} | changed repair schedules: {repaired}")
        for stage in range(3):
            set_stage_budget(self.budget, solve_end, time_end, 4 - stage)
            stats = Counter()
            states, reason = sweep(self.children(state, stage, stats) for state in frontier)
            protected = self.reference[self.tail + stage + 1]
            width = PLATINUM_POOL if stage == 2 else EARLY_WIDTH
            frontier = self.diverse(states, width, lambda state: (
                min(state.collected_materials[0],
                    state.collected_materials[1] if stage == 2 else GOAL, GOAL + 0.05),
                state.remaining_dv / (TANK / 30) - state.current_day,
            ), protected)
            print(f"{('Gold', 'Fuel', 'Platinum')[stage]}: generated {len(states)}, "
                  f"retained {len(frontier)} | {reason} | {dict(stats)}")

        set_stage_budget(self.budget, solve_end, time_end, 1)
        stats = Counter()
        best_by_parent = {}
        completed = 0

        def complete(parent):
            nonlocal completed
            for index, batch in enumerate(self.children(parent, 3, stats), 1):
                for child in batch:
                    completed += 1
                    if parent not in best_by_parent or result_rank(child) > result_rank(best_by_parent[parent]):
                        best_by_parent[parent] = child
                # Sweep consumes batches round-robin, even when no child was feasible.
                yield ()
                if index == NICKEL_PROBES:
                    stats["per-parent probe limit"] += 1
                    break

        _, reason = sweep(complete(parent) for parent in frontier)
        best = max([incumbent, *best_by_parent.values()], key=result_rank)
        print(f"Nickel: platinum parents {len(frontier)}, parents with completions {len(best_by_parent)}, "
              f"complete candidates {completed} | {reason} | {dict(stats)}")
        print(f"Best screened score: {min(best.collected_materials)!r}")
        return best


def run(started, deadline):
    import beam_search as engine

    budget = Budget(engine.pk.lambert_problem, deadline)
    engine.pk.lambert_problem = budget
    try:
        saved = json.loads(SOURCE.read_text(encoding="utf-8"))
        reference, indices, incumbent_minimum, incumbent_score = load_reference(saved, engine)
        budget.solve_limit = MAX_SOLVES - 5 * (len(reference) - 1)
        budget.deadline = deadline - VALIDATION_SECONDS
        shortlist = CatalogShortlist(engine.np, engine.udp, engine.esa["T_START"].mjd2000,
                                     budget.check_time)
        best = CompletionSearch(engine, shortlist, budget, reference, indices).run()
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
        print(f"Candidate official score: {official!r} | independently replayed: {replayed!r}")
        if min(official, replayed) <= BASELINE:
            print("Official validation did not confirm an improvement.")
            report(reference[-1], incumbent_score, incumbent_minimum, budget, started)
            return 0
        budget.check_time()
        try:
            export_result(final, replayed, output=OUTPUT, algorithm="Completion-aware downstream search")
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
