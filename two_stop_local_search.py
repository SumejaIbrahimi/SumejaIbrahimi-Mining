"""Bounded gold/platinum replacement on the catalog-wide incumbent."""

from functools import lru_cache
import json
import math
import subprocess
import sys

from gold_stop_experiment import SOURCE, shortlist


GOLD_SCHEDULES = ((1611, 24), (1613, 22))
PLATINUM_SCHEDULES = ((1650, 25), (1652, 23))
LEG_COUNT = 22
SOLVE_LIMIT = 274
TIME_LIMIT = 30


def platinum_options(udp, visited, gold_id, epoch, mining_days, minimum_yield):
    """Keep 8719 and rank fresh platinum by two-sided arrival-epoch geometry."""
    left_position, left_velocity = udp.asteroids[gold_id].eph(epoch)
    right_position, right_velocity = udp.asteroids[1266].eph(epoch)

    def rank(asteroid_id):
        position, velocity = udp.asteroids[asteroid_id].eph(epoch)
        incoming = math.dist(position, left_position) / (16 * 86400)
        outgoing = math.dist(position, right_position) / (62 * 86400)
        geometry = (incoming + outgoing + math.dist(velocity, left_velocity)
                    + math.dist(velocity, right_velocity))
        mined = min(udp.asteroid_masses[asteroid_id], mining_days / 30)
        return geometry / mined, asteroid_id

    eligible = (
        i for i, (material, mass) in enumerate(zip(
            udp.asteroid_materials_types, udp.asteroid_masses,
        ))
        if i not in visited and i != gold_id and material == 1
        and min(mass, mining_days / 30) > max(0.0, minimum_yield)
    )
    return [8719, *sorted(eligible, key=rank)[:2]]


def run():
    from beam_search import (
        BeamState, RouteStop, apply_candidate, calculate_transfer_dv, esa, udp, validate_route,
    )

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    route = tuple(RouteStop(*stop) for stop in zip(
        saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
    ))
    if len(route) != LEG_COUNT + 1:
        raise ValueError("Expected a 23-stop catalog-wide incumbent.")
    index = saved["visited_asteroids"].index(4088)
    left, old_gold, old_platinum, right = route[index - 1:index + 3]
    if (
        left.asteroid_id, left.arrival_day + left.mining_days, old_gold.asteroid_id,
        old_platinum.asteroid_id, right.asteroid_id, right.arrival_day,
    ) != (831, 1593, 4088, 8719, 1266, 1737):
        raise ValueError("The saved segment does not match the experiment.")

    probes = 0

    @lru_cache(maxsize=None)
    def transfer(source_id, departure, target_id, arrival):
        nonlocal probes
        if probes >= SOLVE_LIMIT - 2 * LEG_COUNT:
            raise RuntimeError("Lambert screening budget exhausted.")
        probes += 1
        return calculate_transfer_dv(source_id, target_id, departure, arrival)

    def advance(state, stop):
        asteroid_id = stop.asteroid_id
        dv = transfer(state.current_id, state.current_day, asteroid_id, stop.arrival_day)
        return apply_candidate(state, {
            "id": asteroid_id,
            "arrival_day": stop.arrival_day,
            "dv": dv,
            "material": int(udp.asteroid_materials_types[asteroid_id]),
            "mass": float(udp.asteroid_masses[asteroid_id]),
        }, stop.mining_days)

    first = route[0]
    states = [BeamState(first.asteroid_id, first.arrival_day + first.mining_days,
                        10000.0, (0.0, 0.0, 0.0), frozenset({first.asteroid_id}), (first,))]
    for stop in route[1:]:
        state = advance(states[-1], stop)
        if state is None:
            raise ValueError(f"Incumbent replay failed at asteroid {stop.asteroid_id}.")
        states.append(state)
    incumbent = states[-1]
    baseline = saved["official_score"]
    if not math.isclose(min(incumbent.collected_materials), baseline, rel_tol=0, abs_tol=1e-8):
        raise ValueError("The incumbent replay does not match its saved score.")

    old_yield = min(udp.asteroid_masses[8719], old_platinum.mining_days / 30)
    minimum_yield = baseline - (incumbent.collected_materials[1] - old_yield)
    epoch = esa["T_START"].mjd2000
    best = None
    screened = feasible = 0
    for gold_day, gold_stay in GOLD_SCHEDULES:
        gold_ids = shortlist(udp, incumbent.visited_asteroids, epoch + gold_day, gold_stay)
        for gold_id in gold_ids[:8]:
            try:
                gold_state = advance(states[index - 1], RouteStop(gold_id, gold_day, gold_stay))
            except (ValueError, RuntimeError):
                continue
            if gold_state is None:
                continue
            for platinum_day, platinum_stay in PLATINUM_SCHEDULES:
                options = platinum_options(
                    udp, incumbent.visited_asteroids, gold_id, epoch + platinum_day,
                    platinum_stay, minimum_yield,
                )
                for platinum_id in options[:3]:
                    screened += 1
                    stop = RouteStop(platinum_id, platinum_day, platinum_stay)
                    state = gold_state
                    try:
                        for next_stop in (stop,) + route[index + 2:]:
                            state = advance(state, next_stop)
                            if state is None:
                                break
                    except (ValueError, RuntimeError):
                        continue
                    if state is None:
                        continue
                    feasible += 1
                    score = min(state.collected_materials)
                    if score > baseline + 1e-8 and (
                        best is None or (score, state.remaining_dv) >
                        (min(best.collected_materials), best.remaining_dv)
                    ):
                        best = state

    print(f"Screened {screened} pairs; {feasible} fuel-feasible; {probes} replay and screening solves.")
    if best is None:
        print("No improving trial reached official validation.")
        return
    if len(best.route) != LEG_COUNT + 1 or probes + 2 * LEG_COUNT > SOLVE_LIMIT:
        raise RuntimeError("Full-route validation would exceed the Lambert budget.")
    score = validate_route(best, require_full_route=True)
    if score <= baseline:
        print(f"Official ESA score {score!r} | incumbent retained")
        return
    gold, platinum = best.route[index:index + 2]
    print(f"Gold {gold.asteroid_id} on day {gold.arrival_day}, stay {gold.mining_days}; "
          f"platinum {platinum.asteroid_id} on day {platinum.arrival_day}, stay {platinum.mining_days}")
    print(f"Official ESA score {score!r} | improvement {score - baseline:+.12f}")


def main():
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", "-c", "from two_stop_local_search import run; run()"],
            timeout=TIME_LIMIT,
        )
    except subprocess.TimeoutExpired:
        print("30-second limit reached. Experiment stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    sys.exit(main())
