"""Reoptimize the four stops after the verified itinerary's day-1670 anchor."""

from collections import defaultdict, deque
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from global_multistart_beam import (
    CatalogShortlist, MAX_SOLVES, MISSION_END, SearchLimit, TANK, TIME_LIMIT,
    VALIDATION_SECONDS,
)


SOURCE = Path(__file__).with_name("global_multistart_best_result.json")
OUTPUT = Path(__file__).with_name("four_stop_suffix_best_result.json")
BASELINE = 8.75296423786425
GOAL = 9.0
ANCHOR = (7230, 1670)
KINDS = frozenset(range(4))
BEAM_WIDTH = 24
PER_MATERIAL = 6
SCORE_TOLERANCE = 1e-8


class Budget:
    """Count actual Lambert calls, including the independent official evaluator."""

    def __init__(self, lambert, deadline):
        self.lambert = lambert
        self.deadline = deadline
        self.solve_limit = MAX_SOLVES
        self.solves = 0

    def check_time(self):
        if time.perf_counter() >= self.deadline:
            raise SearchLimit("time limit reached")

    def __call__(self, *args, **kwargs):
        self.check_time()
        if self.solves >= self.solve_limit:
            raise SearchLimit("Lambert solve limit reached")
        self.solves += 1
        return self.lambert(*args, **kwargs)


def candidate(udp, target, arrival, dv):
    return {
        "id": target,
        "arrival_day": arrival,
        "dv": dv,
        "material": int(udp.asteroid_materials_types[target]),
        "mass": float(udp.asteroid_masses[target]),
    }


def replay(saved, engine):
    """Reconstruct every state from the itinerary with fresh, uncached transfers."""
    ids = saved["visited_asteroids"]
    arrivals = saved["arrival_times"]
    stays = saved["mining_times"]
    if not ids or len(ids) != len(arrivals) or len(ids) != len(stays):
        raise ValueError("Inconsistent itinerary arrays")
    if (arrivals[0], stays[0]) != (0, 1) or len(set(ids)) != len(ids):
        raise ValueError("Invalid itinerary start or repeated asteroid")
    if any(type(target) is not int or not 0 <= target < len(engine.udp.asteroids)
           for target in ids):
        raise ValueError("Invalid asteroid ID")

    state = engine.BeamState(
        ids[0], 1, TANK, (0.0, 0.0, 0.0), frozenset({ids[0]}),
        (engine.RouteStop(ids[0], 0, 1),),
    )
    states = [state]
    minimum = TANK
    for target, arrival, stay in zip(ids[1:], arrivals[1:], stays[1:]):
        if (type(arrival) is not int or type(stay) is not int
                or arrival <= state.current_day or not 1 <= stay <= 60
                or arrival + stay > MISSION_END):
            raise ValueError(f"Invalid schedule at asteroid {target}")
        dv = engine.calculate_transfer_dv(state.current_id, target, state.current_day, arrival)
        minimum = min(minimum, state.remaining_dv - dv)
        state = engine.apply_candidate(state, candidate(engine.udp, target, arrival, dv), stay)
        if state is None:
            raise ValueError(f"Infeasible replay leg to asteroid {target}")
        states.append(state)
    return states, minimum


def check_result(saved, state, official):
    if (not math.isfinite(official)
            or not math.isclose(official, saved["official_score"], rel_tol=0, abs_tol=SCORE_TOLERANCE)
            or not math.isclose(official, saved["score"], rel_tol=0, abs_tol=SCORE_TOLERANCE)
            or state.current_day != saved["final_day"]
            or not math.isclose(state.remaining_dv, saved["remaining_dv"], rel_tol=0, abs_tol=1e-6)
            or any(not math.isclose(total, saved["collected_materials"][str(kind)],
                                    rel_tol=0, abs_tol=SCORE_TOLERANCE)
                   for kind, total in enumerate(state.collected_materials))):
        raise ValueError("Replay does not match the result")


def reconstruct_anchor(saved, engine):
    if saved["official_score"] != BASELINE:
        raise ValueError("Unexpected baseline score")
    states, minimum = replay(saved, engine)
    if len(states) < 5 or (states[-5].current_id, states[-5].current_day) != ANCHOR:
        raise ValueError("Expected asteroid 7230 at day 1670, exactly four stops from the end")
    official = engine.validate_route(states[-1], require_full_route=True)
    check_result(saved, states[-1], official)
    return states[-5], states[-1], minimum, official


def mining_requirements(totals):
    # Use the same addition as the transition, including at exact score boundaries.
    return {**{kind: next((days for days in range(1, 61) if total + days / 30 > GOAL), 61)
               for kind, total in enumerate(totals)}, 3: 1}


def eligible_groups(shortlist, anchor):
    """A material has only one visit, so its deposit must cover the whole deficit."""
    return [
        shortlist.np.asarray([
            target for target in group
            if int(target) not in anchor.visited_asteroids
            and (kind == 3 or anchor.collected_materials[kind] + shortlist.masses[target] > GOAL)
        ], dtype=int)
        for kind, group in enumerate(shortlist.groups)
    ]


def minimum_days(kinds, stays):
    # At least one whole day of flight per remaining stop.
    return sum(1 + stays[kind] for kind in kinds)


def select_frontier(children):
    """Share slots across remaining material sets and preserve timing/fuel variants."""
    groups = defaultdict(list)
    for state, remaining in children:
        groups[remaining].append(state)

    queues = []
    for remaining, states in groups.items():
        ranked = sorted(states, key=lambda state: (
            state.remaining_dv / (TANK / 30) - state.current_day,
            min(state.collected_materials), state.remaining_dv,
        ), reverse=True)
        distinct, seen = deque(), set()
        for state in ranked:
            key = (state.current_id, state.current_day // 3, int(state.remaining_dv // 750))
            if key not in seen:
                seen.add(key)
                distinct.append((state, remaining))
        queues.append(distinct)

    selected = []
    while queues and len(selected) < BEAM_WIDTH:
        for queue in queues:
            selected.append(queue.popleft())
            if len(selected) == BEAM_WIDTH:
                break
        queues = [queue for queue in queues if queue]
    return selected


def search(engine, shortlist, budget, anchor, incumbent):
    @lru_cache(maxsize=MAX_SOLVES)
    def transfer(source, target, departure, arrival):
        try:
            return engine.calculate_transfer_dv(source, target, departure, arrival)
        except (ValueError, RuntimeError) as error:
            if isinstance(error, SearchLimit):
                raise
            return math.nan

    stays = mining_requirements(anchor.collected_materials)
    groups = eligible_groups(shortlist, anchor)
    if (max(stays.values()) > 60 or any(len(group) == 0 for group in groups)
            or anchor.current_day + minimum_days(KINDS, stays) > MISSION_END):
        return None, "material or time bound"
    shortlist.groups = groups
    incumbent_ids = {
        int(engine.udp.asteroid_materials_types[stop.asteroid_id]): stop.asteroid_id
        for stop in incumbent.route[len(anchor.route):]
    }
    frontier = [(anchor, KINDS)]
    best = None
    try:
        for _ in range(4):
            children = []
            for parent, remaining in frontier:
                for kind in sorted(remaining):
                    later = remaining - {kind}
                    latest_departure = MISSION_END - minimum_days(later, stays)
                    for arrival in range(parent.current_day + 1, latest_departure - stays[kind] + 1):
                        budget.check_time()
                        targets = shortlist(parent, arrival, kinds=(kind,), per_material=PER_MATERIAL)
                        incumbent_id = incumbent_ids.get(kind)
                        if incumbent_id is not None and incumbent_id not in targets:
                            if incumbent_id in groups[kind] and incumbent_id not in parent.visited_asteroids:
                                targets.append(incumbent_id)
                        for target in targets:
                            dv = transfer(parent.current_id, target, parent.current_day, arrival)
                            if not 0 <= dv <= parent.remaining_dv:
                                continue
                            option = candidate(engine.udp, target, arrival, dv)
                            maximum = min(60, latest_departure - arrival)
                            # At the last stop a longer stay cannot worsen the score or fuel.
                            durations = range(stays[kind], maximum + 1) if later else (maximum,)
                            for duration in durations:
                                child = engine.apply_candidate(parent, option, duration)
                                if child is None:
                                    continue
                                if later:
                                    children.append((child, later))
                                elif min(child.collected_materials) > GOAL:
                                    if best is None or result_rank(child) > result_rank(best):
                                        best = child
            budget.check_time()
            if not children:
                break
            frontier = select_frontier(children)
    except SearchLimit as error:
        return best, str(error)
    return best, "suffix search finished"


def result_rank(state):
    return min(state.collected_materials), state.remaining_dv, -state.current_day


def report(state, score, minimum, budget, started):
    print(f"Official score: {score!r} | improvement: {score - BASELINE:+.15f}")
    print(f"Material totals (IDs 0, 1, 2): {state.collected_materials}")
    print(f"Final day: {state.current_day} | remaining fuel: {state.remaining_dv:.6f} m/s"
          f" | minimum fuel immediately after a transfer: {minimum:.6f} m/s")
    print(f"Lambert solves: {budget.solves} | runtime: {time.perf_counter() - started:.2f} s")


def run(started, deadline):
    import beam_search as engine

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    budget = Budget(engine.pk.lambert_problem, deadline)
    engine.pk.lambert_problem = budget
    try:
        anchor, incumbent, baseline_minimum, baseline_score = reconstruct_anchor(saved, engine)
        print(f"Verified anchor: asteroid {anchor.current_id}, day {anchor.current_day}, "
              f"fuel {anchor.remaining_dv:.6f} m/s, materials {anchor.collected_materials}")
        # Each official validation solves every leg twice. The JSON replay adds one pass.
        budget.solve_limit = MAX_SOLVES - 5 * (len(incumbent.route) - 1)
        budget.deadline = deadline - VALIDATION_SECONDS
        shortlist = CatalogShortlist(engine.np, engine.udp, engine.esa["T_START"].mjd2000,
                                     budget.check_time)
        best, reason = search(engine, shortlist, budget, anchor, incumbent)
        budget.solve_limit, budget.deadline = MAX_SOLVES, deadline
        print(reason)
        if best is None:
            print("No suffix screened above 9.0. Reporting the verified baseline.")
            report(incumbent, baseline_score, baseline_minimum, budget, started)
            return 0

        official = engine.validate_route(best, require_full_route=True)
        if not math.isfinite(official) or official <= BASELINE:
            raise ValueError("Candidate did not improve the official baseline")
        result = {
            "algorithm": "Four-stop suffix search",
            "candidate_strategy": "catalog-wide with deposit filtering",
            "score": min(best.collected_materials),
            "official_score": official,
            "visited_asteroids": [stop.asteroid_id for stop in best.route],
            "arrival_times": [stop.arrival_day for stop in best.route],
            "mining_times": [stop.mining_days for stop in best.route],
            "collected_materials": dict(enumerate(best.collected_materials)),
            "remaining_dv": best.remaining_dv,
            "final_day": best.current_day,
        }
        # Replay the JSON route independently of the search states and transfer cache.
        payload = json.dumps(result, indent=4, allow_nan=False) + "\n"
        serialized = json.loads(payload)
        states, minimum = replay(serialized, engine)
        final = states[-1]
        if final.route[:len(anchor.route)] != anchor.route:
            raise ValueError("Replayed route changed the verified prefix")
        replayed_score = engine.validate_route(final, require_full_route=True)
        check_result(serialized, final, replayed_score)
        if replayed_score <= BASELINE:
            raise ValueError("Independent replay did not improve the official baseline")
        budget.check_time()
        try:
            with OUTPUT.open("x", encoding="utf-8") as file:
                file.write(payload)
        except FileExistsError:
            print(f"{OUTPUT.name} already exists and was preserved.")
        else:
            print(f"Saved {OUTPUT.name}")
        report(final, official, minimum, budget, started)
        return 0
    except (SearchLimit, ValueError) as error:
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
