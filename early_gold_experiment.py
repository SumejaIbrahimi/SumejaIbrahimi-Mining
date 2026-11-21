"""Bounded gold replacement between asteroid 410 and the refuel at 1850."""

from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys


SOURCE = Path("catalog_wide_best_result.json")
SCHEDULES = ((205, 29), (210, 27))
LEG_COUNT = 22
SOLVE_LIMIT = 98
TIME_LIMIT = 30
TANK = 10000.0


def gold_candidates(udp, visited, epoch, mining_days):
    position, velocity = udp.asteroids[1444].eph(epoch)

    def rank(asteroid_id):
        target_position, target_velocity = udp.asteroids[asteroid_id].eph(epoch)
        return (
            math.hypot(math.dist(position, target_position) / (80 * 86400),
                       math.dist(velocity, target_velocity)),
            asteroid_id,
        )

    eligible = (
        i for i, (material, mass) in enumerate(zip(
            udp.asteroid_materials_types, udp.asteroid_masses,
        ))
        if i not in visited and material == 0 and min(mass, mining_days / 30) > 0.728173
    )
    return sorted(eligible, key=rank)[:8]


def run():
    from beam_search import (
        BeamState, RouteStop, apply_candidate, calculate_transfer_dv, esa, udp, validate_route,
    )

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    route = tuple(RouteStop(*stop) for stop in zip(
        saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
    ))
    baseline = saved["official_score"]
    if len(route) != LEG_COUNT + 1 or baseline != 4.151473504756365 or route[1:5] != (
        RouteStop(410, 101, 29), RouteStop(1444, 210, 22),
        RouteStop(772, 292, 10), RouteStop(1850, 382, 24),
    ):
        raise ValueError("The saved incumbent does not match the expected segment.")

    probes = 0

    @lru_cache(maxsize=None)
    def transfer(source_id, departure_day, target_id, arrival_day):
        nonlocal probes
        if probes >= SOLVE_LIMIT - 2 * LEG_COUNT:
            raise RuntimeError("Lambert screening budget exhausted.")
        probes += 1
        return calculate_transfer_dv(
            start_id=source_id, target_id=target_id,
            departure_day=departure_day, arrival_day=arrival_day,
        )

    def advance(state, stop):
        asteroid_id = stop.asteroid_id
        leg = f"{state.current_id} -> {asteroid_id}"
        try:
            dv = transfer(state.current_id, state.current_day, asteroid_id, stop.arrival_day)
        except (RuntimeError, ValueError) as error:
            return None, f"{leg}: Lambert solver error ({error}); fuel margin unavailable"
        if not math.isfinite(dv) or dv < 0:
            return None, f"{leg}: invalid Lambert delta-v ({dv}); fuel margin unavailable"
        margin = state.remaining_dv - dv
        if margin < 0:
            return None, f"{leg}: insufficient fuel (delta-v {dv:.2f} m/s, margin {margin:+.2f} m/s)"
        material = int(udp.asteroid_materials_types[asteroid_id])
        mass = float(udp.asteroid_masses[asteroid_id])
        next_state = apply_candidate(state, {
            "id": asteroid_id, "arrival_day": stop.arrival_day,
            "dv": dv, "material": material, "mass": mass,
        }, stop.mining_days)
        if next_state is None:
            return None, f"{leg}: route constraint failed (fuel margin {margin:+.2f} m/s)"
        if asteroid_id == 1850 and next_state.remaining_dv != TANK:
            refill_margin = next_state.remaining_dv - TANK
            return None, (f"{leg}: refuel did not restore a full tank "
                          f"(arrival margin {margin:+.2f} m/s, tank margin {refill_margin:+.2f} m/s)")
        return next_state, None

    first = route[0]
    states = [BeamState(first.asteroid_id, first.arrival_day + first.mining_days,
                        TANK, (0.0, 0.0, 0.0), frozenset({first.asteroid_id}), (first,))]
    for stop in route[1:]:
        state, failure = advance(states[-1], stop)
        if failure:
            raise ValueError(f"Incumbent replay failed: {failure}")
        states.append(state)
    incumbent = states[-1]
    if not math.isclose(min(incumbent.collected_materials), baseline, rel_tol=0, abs_tol=1e-8):
        raise ValueError("The incumbent replay does not match the saved score.")

    epoch = esa["T_START"].mjd2000
    platinum = RouteStop(772, 294, 8)
    best = None
    for day, stay in SCHEDULES:
        candidates = gold_candidates(udp, incumbent.visited_asteroids, epoch + day, stay)
        for asteroid_id in candidates[:8]:
            label = f"Gold {asteroid_id} on day {day}, stay {stay}"
            state = states[1]
            for stop in (RouteStop(asteroid_id, day, stay), platinum, *route[4:]):
                state, failure = advance(state, stop)
                if failure:
                    print(f"{label} | rejected at {failure}")
                    break
            else:
                score = min(state.collected_materials)
                print(f"{label} | feasible | screening score {score:.9f}")
                if score > baseline + 1e-8 and (best is None or score > min(best.collected_materials)):
                    best = state

    print(f"Replay and screening Lambert solves: {probes}")
    if best is None:
        print("No improving trial reached official validation.")
        return
    if len(best.route) != LEG_COUNT + 1 or probes + 2 * LEG_COUNT > SOLVE_LIMIT:
        raise RuntimeError("Full-route validation would exceed the Lambert budget.")
    score = validate_route(best, require_full_route=True)
    gold = best.route[2]
    print(f"Gold {gold.asteroid_id} on day {gold.arrival_day}, stay {gold.mining_days}"
          f" | official ESA score {score!r} | delta vs incumbent {score - baseline:+.12f}")


def main():
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", "-c", "from early_gold_experiment import run; run()"],
            timeout=TIME_LIMIT,
        )
    except subprocess.TimeoutExpired:
        print("30-second limit reached. Experiment stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    sys.exit(main())
