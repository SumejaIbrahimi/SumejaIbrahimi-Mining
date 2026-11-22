"""Bounded gold/platinum replacement and retiming before the day-406 departure."""

from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from early_gold_experiment import export_result


SOURCE = Path("middle_gold_best_result.json")
OUTPUT = Path("anchored_two_stop_best_result.json")
INCUMBENT_SCORE = 4.320457054847707
TARGET_SCORE = 4.370457054847707
GOLD_ARRIVALS = (204, 207, 210, 213)
GOLD_STAYS = (27, 29, 30, 31)
PLATINUM_ARRIVALS = (286, 290, 294, 298)
PLATINUM_STAYS = (6, 8, 10, 12)
FUEL_ARRIVALS = (378, 382, 386, 390)
LEG_COUNT = 22
SOLVE_LIMIT = 2500
TIME_LIMIT = 60
VALIDATION_SOLVES = 2 * LEG_COUNT
TOLERANCE = 1e-8


class BudgetExceeded(RuntimeError):
    pass


class TransferCache:
    def __init__(self, calculate, deadline):
        self.deadline = deadline
        self.solves = 0
        self.transfer = lru_cache(maxsize=None)(self._solve)
        self.calculate = calculate

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("60-second wall-time limit reached.")

    def _solve(self, source, departure, target, arrival):
        self.check_time()
        if self.solves >= SOLVE_LIMIT - VALIDATION_SOLVES:
            raise BudgetExceeded("Lambert solve budget exhausted.")
        self.solves += 1
        try:
            return self.calculate(source, target, departure, arrival)
        except (RuntimeError, ValueError):
            # Cache failed solves too. apply_candidate rejects non-finite costs.
            return math.nan


def shortlist(udp, excluded, epoch, material, incumbent_id, neighbors,
              arrivals, max_stay, minimum_yield, limit):
    """Keep the incumbent and rank eligible fresh IDs by two-sided geometry."""
    references = [
        (epoch + day, tuple((udp.asteroids[i].eph(epoch + day), flight_days)
                            for i, flight_days in neighbors))
        for day in arrivals
    ]

    def priority(asteroid_id):
        cost = 0.0
        for date, endpoints in references:
            position, velocity = udp.asteroids[asteroid_id].eph(date)
            for (other_position, other_velocity), flight_days in endpoints:
                cost += math.hypot(
                    math.dist(position, other_position) / (flight_days * 86400),
                    math.dist(velocity, other_velocity),
                )
        return cost, asteroid_id

    eligible = (
        i for i, kind in enumerate(udp.asteroid_materials_types)
        if kind == material and i not in excluded
        and min(udp.asteroid_masses[i], max_stay / 30) >= minimum_yield - TOLERANCE
    )
    return [incumbent_id, *sorted(eligible, key=priority)[:limit - 1]]


def route_rank(state):
    return min(state.collected_materials), state.remaining_dv


def search_window(prefix, suffix, stop_type, gold_ids, platinum_ids, advance,
                  yield_for, minimum_yields, check_time):
    best = None
    rejected = Counter()
    for gold_id in gold_ids:
        for gold_day in GOLD_ARRIVALS:
            for gold_stay in GOLD_STAYS:
                check_time()
                if yield_for(gold_id, gold_stay) < minimum_yields[0] - TOLERANCE:
                    continue
                gold = advance(prefix, stop_type(gold_id, gold_day, gold_stay))
                if gold is None:
                    rejected["incoming gold leg"] += 1
                    continue
                for platinum_id in platinum_ids:
                    for platinum_day in PLATINUM_ARRIVALS:
                        for platinum_stay in PLATINUM_STAYS:
                            check_time()
                            if yield_for(platinum_id, platinum_stay) < minimum_yields[1] - TOLERANCE:
                                continue
                            platinum = advance(gold, stop_type(platinum_id, platinum_day, platinum_stay))
                            if platinum is None:
                                rejected["gold to platinum"] += 1
                                continue
                            for fuel_day in FUEL_ARRIVALS:
                                fuel = advance(platinum, stop_type(1850, fuel_day, 406 - fuel_day))
                                if fuel is None:
                                    rejected["platinum to refuel"] += 1
                                    continue
                                state = fuel
                                for stop in suffix:
                                    state = advance(state, stop)
                                    if state is None:
                                        rejected[f"suffix at {stop.asteroid_id}"] += 1
                                        break
                                if state is not None and (best is None or route_rank(state) > route_rank(best)):
                                    best = state
    return best, rejected


def run(deadline):
    from beam_search import (
        BeamState, RouteStop, apply_candidate, calculate_transfer_dv, esa, udp, validate_route,
    )

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    route = tuple(RouteStop(*fields) for fields in zip(
        saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
    ))
    if (saved["official_score"] != INCUMBENT_SCORE or len(route) != LEG_COUNT + 1
            or route[1:5] != (RouteStop(410, 101, 29), RouteStop(4745, 210, 27),
                              RouteStop(772, 294, 8), RouteStop(1850, 382, 24))):
        raise ValueError("Saved incumbent does not match the anchored segment.")

    transfers = TransferCache(calculate_transfer_dv, deadline)

    def yield_for(asteroid_id, stay):
        return min(float(udp.asteroid_masses[asteroid_id]), stay / 30)

    def advance(state, stop):
        asteroid_id = stop.asteroid_id
        if (asteroid_id in state.visited_asteroids or stop.arrival_day - state.current_day < 0.1
                or not 1 <= stop.mining_days <= 60 or stop.arrival_day + stop.mining_days > 1827):
            return None
        dv = transfers.transfer(state.current_id, state.current_day, asteroid_id, stop.arrival_day)
        return apply_candidate(state, {
            "id": asteroid_id,
            "arrival_day": stop.arrival_day,
            "dv": dv,
            "material": int(udp.asteroid_materials_types[asteroid_id]),
            "mass": float(udp.asteroid_masses[asteroid_id]),
        }, stop.mining_days)

    first = route[0]
    state = BeamState(first.asteroid_id, first.arrival_day + first.mining_days,
                      10000.0, (0.0, 0.0, 0.0), frozenset({first.asteroid_id}), (first,))
    for index, stop in enumerate(route[1:], start=1):
        state = advance(state, stop)
        if state is None:
            raise ValueError(f"Incumbent replay failed at asteroid {stop.asteroid_id}.")
        if index == 1:
            prefix = state
    if not math.isclose(route_rank(state)[0], INCUMBENT_SCORE, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError("Incumbent replay score differs from the saved official score.")

    # Keep below-target options eligible so a smaller official improvement can be saved.
    minimum_yields = tuple(
        INCUMBENT_SCORE - (state.collected_materials[material] - yield_for(stop.asteroid_id, stop.mining_days))
        for material, stop in enumerate(route[2:4])
    )
    epoch = esa["T_START"].mjd2000
    excluded = state.visited_asteroids
    gold_ids = shortlist(udp, excluded, epoch, 0, 4745, ((410, 80), (772, 57)),
                         GOLD_ARRIVALS, max(GOLD_STAYS), minimum_yields[0], 6)
    platinum_ids = shortlist(udp, excluded, epoch, 1, 772, ((4745, 57), (1850, 80)),
                             PLATINUM_ARRIVALS, max(PLATINUM_STAYS), minimum_yields[1], 5)
    print(f"Gold shortlist {gold_ids} | platinum shortlist {platinum_ids}")
    best, rejected = search_window(prefix, route[5:], RouteStop, gold_ids, platinum_ids,
                                   advance, yield_for, minimum_yields, transfers.check_time)
    print(f"Replay and screening: {transfers.solves} Lambert solves | rejections {dict(rejected)}")
    if best is None or route_rank(best)[0] <= INCUMBENT_SCORE:
        print("No improving feasible route found. No official validation used.")
        return

    print(f"Best screened score {route_rank(best)[0]:.15f} | changed stops {best.route[2:5]}")
    transfers.check_time()
    score = validate_route(best, require_full_route=True)
    transfers.check_time()
    print(f"Official ESA score {score!r} | improvement {score - INCUMBENT_SCORE:+.12f}"
          f" | target {'met' if score >= TARGET_SCORE else 'not met'}")
    if score > INCUMBENT_SCORE:
        try:
            export_result(best, score, output=OUTPUT, algorithm="Anchored two-stop search")
        except FileExistsError:
            print(f"{OUTPUT} already exists. Kept the existing result.")
        else:
            print(f"Saved {OUTPUT}")


def main():
    deadline = time.monotonic() + TIME_LIMIT
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", "-c",
             "import sys; from anchored_two_stop_search import run; run(float(sys.argv[1]))",
             str(deadline)],
            timeout=max(0.0, deadline - time.monotonic()),
        )
    except subprocess.TimeoutExpired:
        print("60-second wall-time limit reached. Worker stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    sys.exit(main())
