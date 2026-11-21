"""Bounded replacement of asteroid 4088 on the catalog-wide route."""

import json
import math
from pathlib import Path


SOURCE = Path("catalog_wide_best_result.json")
SCHEDULES = ((1613, 17), (1611, 19), (1611, 21), (1613, 21))
CANDIDATE_LIMIT = 8
LEG_COUNT = 22
SOLVE_LIMIT = 130


def shortlist(udp, visited, epoch, mining_days):
    position, velocity = udp.asteroids[4088].eph(epoch)
    old_yield = udp.asteroid_masses[4088]
    eligible = [
        i for i, (material, mass) in enumerate(zip(
            udp.asteroid_materials_types, udp.asteroid_masses,
        ))
        if i not in visited and material == 0 and min(mass, mining_days / 30) > old_yield
    ]

    def proximity(asteroid_id):
        target_position, target_velocity = udp.asteroids[asteroid_id].eph(epoch)
        return (
            math.hypot(
                math.dist(position, target_position) / (20 * 86400),
                math.dist(velocity, target_velocity),
            ),
            asteroid_id,
        )

    ranked = sorted(eligible, key=proximity)
    nearby = ranked[:CANDIDATE_LIMIT - 2]
    richer = sorted(ranked[len(nearby):], key=lambda i: (-udp.asteroid_masses[i], i))
    return nearby + richer[:2]


def main():
    from beam_search import (
        BeamState, RouteStop, apply_candidate, calculate_transfer_dv, esa, udp, validate_route,
    )

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    original = tuple(RouteStop(*stop) for stop in zip(
        saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
    ))
    if len(original) != LEG_COUNT + 1:
        raise ValueError("Expected the 23-stop catalog-wide route.")
    index = saved["visited_asteroids"].index(4088)
    previous, gold, following = original[index - 1:index + 2]
    if previous != RouteStop(831, 1565, 28) or gold != RouteStop(4088, 1613, 17) or (
        following.asteroid_id, following.arrival_day
    ) != (8719, 1650):
        raise ValueError("The saved schedule around asteroid 4088 has changed.")

    probes = 0

    def transfer(source, target):
        nonlocal probes
        # The existing validator and ESA fitness each solve the full route once.
        if probes >= SOLVE_LIMIT - 2 * LEG_COUNT:
            raise RuntimeError("Lambert screening budget exhausted.")
        probes += 1
        return calculate_transfer_dv(
            source.asteroid_id, target.asteroid_id,
            source.arrival_day + source.mining_days, target.arrival_day,
        )

    def replay_stop(state, stop, dv):
        asteroid_id = stop.asteroid_id
        return apply_candidate(state, {
            "id": asteroid_id,
            "arrival_day": stop.arrival_day,
            "dv": dv,
            "material": int(udp.asteroid_materials_types[asteroid_id]),
            "mass": float(udp.asteroid_masses[asteroid_id]),
        }, stop.mining_days)

    first = original[0]
    states = [BeamState(first.asteroid_id, first.arrival_day + first.mining_days,
                        10000.0, (0.0, 0.0, 0.0), frozenset({first.asteroid_id}), (first,))]
    saved_dv = []
    for source, target in zip(original, original[1:]):
        dv = transfer(source, target)
        state = replay_stop(states[-1], target, dv)
        if state is None:
            raise ValueError(f"Incumbent replay failed at asteroid {target.asteroid_id}.")
        saved_dv.append(dv)
        states.append(state)

    best = None
    incumbent_score = saved["official_score"]
    for arrival_day, mining_days in SCHEDULES:
        candidates = shortlist(
            udp, states[-1].visited_asteroids, esa["T_START"].mjd2000 + arrival_day, mining_days,
        )[:CANDIDATE_LIMIT]
        print(f"Arrival day {arrival_day}, mining {mining_days} days | candidates {candidates}")
        for asteroid_id in candidates:
            stop = RouteStop(asteroid_id, arrival_day, mining_days)
            try:
                inbound = transfer(previous, stop)
                outbound = transfer(stop, following)
            except (RuntimeError, ValueError) as error:
                print(f"Candidate {asteroid_id} | transfer failed ({error})")
                continue
            label = f"Candidate {asteroid_id} | {inbound:.2f} / {outbound:.2f} m/s"
            state = states[index - 1]
            tail = (stop,) + original[index + 1:]
            for target, dv in zip(tail, [inbound, outbound, *saved_dv[index + 1:]], strict=True):
                state = replay_stop(state, target, dv)
                if state is None:
                    print(f"{label} | infeasible at asteroid {target.asteroid_id}")
                    break
            else:
                print(f"{label} | fuel feasible | {state.remaining_dv:.2f} m/s at finish")
                score = min(state.collected_materials)
                if score > incumbent_score and (
                    best is None or (score, state.remaining_dv) >
                    (min(best.collected_materials), best.remaining_dv)
                ):
                    best = state

    print(f"Replay and screening Lambert probes: {probes}")
    if best is None:
        print("No fuel-feasible improvement. No trial submitted for official validation.")
        return
    if len(best.route) != LEG_COUNT + 1 or probes + 2 * LEG_COUNT > SOLVE_LIMIT:
        raise RuntimeError("Full-route validation would exceed the Lambert budget.")
    score = validate_route(best, require_full_route=True)
    stop = best.route[index]
    print(f"Asteroid {stop.asteroid_id}, arrival day {stop.arrival_day}, mining {stop.mining_days} days"
          f" | official ESA score {score!r}")


if __name__ == "__main__":
    main()
