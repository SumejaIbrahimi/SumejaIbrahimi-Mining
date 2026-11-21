"""Bounded replacement of gold asteroid 962 on the early-gold incumbent."""

from functools import lru_cache
import json
import math
from pathlib import Path
import subprocess
import sys


SOURCE = Path("early_gold_best_result.json")
OUTPUT = Path("middle_gold_best_result.json")
SCHEDULES = ((604, 30), (605, 29), (606, 28))
CANDIDATE_LIMIT = 12
LEG_COUNT = 22
SOLVE_LIMIT = 138
TIME_LIMIT = 30


def gold_candidates(udp, visited, arrival_epoch, departure_epoch, stay):
    epochs = (arrival_epoch, departure_epoch)
    reference = [udp.asteroids[962].eph(epoch) for epoch in epochs]
    masses = udp.asteroid_masses

    def rank(asteroid_id):
        geometry = 0
        for epoch, (position, velocity) in zip(epochs, reference):
            target_position, target_velocity = udp.asteroids[asteroid_id].eph(epoch)
            geometry += math.hypot(math.dist(position, target_position) / (60 * 86400),
                                   math.dist(velocity, target_velocity))
        return geometry, -masses[asteroid_id], asteroid_id

    ranked = sorted(
        (i for i, material in enumerate(udp.asteroid_materials_types)
         if i not in visited and material == 0 and min(masses[i], stay / 30) > masses[962]),
        key=rank,
    )
    nearby = ranked[:9]
    richer = sorted(ranked[len(nearby):], key=lambda i: (-min(masses[i], stay / 30), i))
    return nearby + richer[:3]


def export_result(state, score):
    result = {
        "algorithm": "Middle gold local search",
        "candidate_strategy": "catalog-wide",
        "score": min(state.collected_materials),
        "official_score": score,
        "visited_asteroids": [stop.asteroid_id for stop in state.route],
        "arrival_times": [stop.arrival_day for stop in state.route],
        "mining_times": [stop.mining_days for stop in state.route],
        "collected_materials": dict(enumerate(state.collected_materials)),
        "remaining_dv": state.remaining_dv,
        "final_day": state.current_day,
    }
    payload = json.dumps(result, indent=4, allow_nan=False)
    with OUTPUT.open("x", encoding="utf-8") as file:
        file.write(payload + "\n")


def run():
    from beam_search import (
        BeamState, RouteStop, apply_candidate, calculate_transfer_dv, esa, udp, validate_route,
    )

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    route = tuple(RouteStop(*stop) for stop in zip(
        saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
    ))
    baseline = saved["official_score"]
    index = saved["visited_asteroids"].index(962)
    previous, _, following = route[index - 1:index + 2]
    if len(route) != LEG_COUNT + 1 or baseline != 4.27907823625441 or (
        previous.asteroid_id, previous.arrival_day + previous.mining_days,
        following.asteroid_id, following.arrival_day,
    ) != (13, 527, 276, 694):
        raise ValueError("The saved incumbent does not match the expected route.")

    probes = 0

    @lru_cache(maxsize=None)
    def transfer(source, departure, target, arrival):
        nonlocal probes
        if probes >= SOLVE_LIMIT - 2 * LEG_COUNT:
            raise RuntimeError("Lambert screening budget exhausted.")
        probes += 1
        return calculate_transfer_dv(start_id=source, target_id=target,
                                     departure_day=departure, arrival_day=arrival)

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
        updated = apply_candidate(state, {
            "id": asteroid_id,
            "arrival_day": stop.arrival_day,
            "dv": dv,
            "material": int(udp.asteroid_materials_types[asteroid_id]),
            "mass": float(udp.asteroid_masses[asteroid_id]),
        }, stop.mining_days)
        if updated is None:
            return None, f"{leg}: route constraint failed (fuel margin {margin:+.2f} m/s)"
        return updated, None

    first = route[0]
    states = [BeamState(first.asteroid_id, first.arrival_day + first.mining_days,
                        10000.0, (0.0, 0.0, 0.0), frozenset({first.asteroid_id}), (first,))]
    for stop in route[1:]:
        state, failure = advance(states[-1], stop)
        if failure:
            raise ValueError(f"Incumbent replay failed: {failure}")
        states.append(state)
    incumbent = states[-1]
    if not math.isclose(min(incumbent.collected_materials), baseline, rel_tol=0, abs_tol=1e-8):
        raise ValueError("The incumbent replay does not match the saved score.")

    epoch = esa["T_START"].mjd2000
    best = None
    for day, stay in SCHEDULES:
        candidates = gold_candidates(udp, incumbent.visited_asteroids, epoch + day, epoch + 634, stay)
        for asteroid_id in candidates[:CANDIDATE_LIMIT]:
            label = f"Gold {asteroid_id} on day {day}, stay {stay}"
            state = states[index - 1]
            for stop in (RouteStop(asteroid_id, day, stay), *route[index + 1:]):
                state, failure = advance(state, stop)
                if failure:
                    print(f"{label} | rejected at {failure}")
                    break
            else:
                score = min(state.collected_materials)
                print(f"{label} | feasible | {state.remaining_dv:.2f} m/s at finish"
                      f" | screening score {score:.9f}")
                if score > baseline + 1e-8 and (
                    best is None or (score, state.remaining_dv) >
                    (min(best.collected_materials), best.remaining_dv)
                ):
                    best = state

    print(f"Replay and screening Lambert solves: {probes}")
    if best is None:
        print("No improving trial reached official validation.")
        return
    if len(best.route) != LEG_COUNT + 1 or probes + 2 * LEG_COUNT > SOLVE_LIMIT:
        raise RuntimeError("Full-route validation would exceed the Lambert budget.")
    score = validate_route(best, require_full_route=True)
    stop = best.route[index]
    print(f"Gold {stop.asteroid_id} on day {stop.arrival_day}, stay {stop.mining_days}"
          f" | official ESA score {score!r} | delta vs incumbent {score - baseline:+.12f}")
    if score > baseline:
        try:
            export_result(best, score)
        except FileExistsError:
            print(f"{OUTPUT} already exists. Kept the existing result.")
        else:
            print(f"Saved {OUTPUT}")


def main():
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", "-c", "from middle_gold_experiment import run; run()"],
            timeout=TIME_LIMIT,
        )
    except subprocess.TimeoutExpired:
        print("30-second limit reached. Experiment stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    sys.exit(main())
