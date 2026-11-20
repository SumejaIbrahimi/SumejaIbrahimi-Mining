"""Search exactly three ore stops after the saved route's stop at asteroid 332."""

import argparse
from collections import Counter, deque
from functools import lru_cache
from itertools import permutations
import math
from operator import index

from beam_search import apply_candidate, calculate_transfer_dv, load_and_replay_baseline
from beam_search import validate_route
from find_reachable import esa, udp
from mission import calculate_mining_days


SAVED_SCORE = 3.846180920595038
SCORE_TOLERANCE = 1e-8
TRANSFER_DAYS = (20, 40, 60, 80, 100)
TRANSFER_CACHE_SIZE = 65_536


def minimum_mining_days(amount, target, mining_rate_days):
    days = max(1, math.floor((target - amount) * mining_rate_days))
    while amount + days / mining_rate_days <= target:
        days += 1
    return days


def scan_order(
    state, order, deposits, min_days, deadline, transfer_dv, depth_stats,
    *, flight_durations=TRANSFER_DAYS,
):
    """Yield once per transfer probe, with any complete routes from that probe."""
    material, *remaining_order = order
    stats = depth_stats[len(depth_stats) - len(order)]
    minimum_stay = min_days[material]
    remaining_time = sum(min_days[ore] + TRANSFER_DAYS[0] for ore in remaining_order)

    for flight_days in flight_durations:
        arrival_day = state.current_day + flight_days
        latest_stay = math.floor(deadline - arrival_day - remaining_time)
        if minimum_stay > latest_stay:
            stats["time_cutoffs"] += 1
            break

        for asteroid_id, mass, full_stay in deposits[material]:
            if asteroid_id in state.visited_asteroids:
                stats["visited_skips"] += 1
                continue

            stats["probes"] += 1
            dv = transfer_dv(
                index(state.current_id), float(state.current_day),
                index(asteroid_id), float(arrival_day),
            )
            if not math.isfinite(dv) or not 0 <= dv <= state.remaining_dv:
                if not math.isfinite(dv) or dv < 0:
                    stats["lambert_failures"] += 1
                else:
                    stats["fuel_failures"] += 1
                yield ()
                continue

            candidate = {
                "id": asteroid_id,
                "arrival_day": arrival_day,
                "dv": dv,
                "material": material,
                "mass": mass,
            }
            stays = [minimum_stay]
            if minimum_stay < full_stay <= latest_stay:
                stays.append(full_stay)
            if minimum_stay < full_stay and full_stay > latest_stay:
                stats["full_stay_time_skips"] += 1
            children = []
            for mining_days in stays:
                child = apply_candidate(state, candidate, mining_days)
                if child is not None:
                    children.append(child)
                    stats["feasible_children"] += 1
                else:
                    stats["transition_rejections"] += 1

            if remaining_order:
                yield ()
                for child in children:
                    yield from scan_order(
                        child, remaining_order, deposits, min_days, deadline,
                        transfer_dv, depth_stats,
                    )
            else:
                yield children


def scan_first_leg_durations(seed, order, deposits, min_days, deadline, transfer_dv,
                             probes_by_duration, depth_stats):
    """Round-robin independent depth-first searches fixed to one first-leg duration."""
    active = deque(
        (
            days,
            scan_order(
                seed, order, deposits, min_days, deadline, transfer_dv, depth_stats,
                flight_durations=(days,),
            ),
        )
        for days in TRANSFER_DAYS
    )
    while active:
        days, scan = active.popleft()
        children = next(scan, None)
        if children is None:
            continue
        probes_by_duration[days] += 1
        active.append((days, scan))
        yield children


def run_three_stop_tail_search(limit=2000, max_lambert_solves=320_000):
    """Return the saved incumbent or a better, fully validated BeamState."""
    if limit < 1 or max_lambert_solves < 1:
        raise ValueError("The catalog limit and Lambert budget must be positive.")

    states = load_and_replay_baseline("beam_best_result.json")
    if len(states) <= 17:
        raise ValueError("The saved winner is missing the three-stop search seed.")
    seed, incumbent = states[17], states[-1]
    if seed.current_id != 332 or seed.current_day != 1606:
        raise ValueError("Expected asteroid 332, ready to depart on day 1606.")
    official_score = validate_route(incumbent, require_full_route=True)
    if not math.isclose(official_score, SAVED_SCORE, rel_tol=0, abs_tol=SCORE_TOLERANCE):
        raise ValueError("The saved winner's official score differs from the expected score.")

    target = SAVED_SCORE + SCORE_TOLERANCE
    deadline = udp.mission_window[1]
    mining_rate_days = esa["TIME_TO_MINE_FULLY"]
    min_days = tuple(
        minimum_mining_days(amount, target, mining_rate_days)
        for amount in seed.collected_materials
    )
    deposits = [[], [], []]
    for asteroid_id in range(len(udp.asteroids)):
        material = int(udp.asteroid_materials_types[asteroid_id])
        if material == 3 or asteroid_id in seed.visited_asteroids:
            continue
        # Platinum is scarce, so scan its deposits beyond the Gold and Nickel limit.
        if material != 1 and asteroid_id >= limit:
            continue
        mass = float(udp.asteroid_masses[asteroid_id])
        if seed.collected_materials[material] + mass <= target:
            continue
        full_stay = calculate_mining_days({"mass": mass}, partial_mining=False)
        deposits[material].append((asteroid_id, mass, full_stay))

    for ore_deposits in deposits:
        ore_deposits.sort(key=lambda deposit: (-deposit[1], deposit[0]))

    # The catalog, gravitational parameter, and Lambert settings are fixed per run.
    @lru_cache(maxsize=TRANSFER_CACHE_SIZE)
    def transfer_dv(source_id, departure_day, target_id, arrival_day):
        try:
            dv = calculate_transfer_dv(
                start_id=source_id,
                target_id=target_id,
                departure_day=departure_day,
                arrival_day=arrival_day,
            )
        except (RuntimeError, ValueError):
            return math.inf
        return dv if math.isfinite(dv) and dv >= 0 else math.inf

    # Take one probe per turn, including hits, to preserve round-robin scheduling.
    orders = list(permutations(range(3))) if all(deposits) else []
    scans_by_order = dict.fromkeys(orders, 0)
    probes_by_duration = dict.fromkeys(TRANSFER_DAYS, 0)
    depth_stats = (Counter(), Counter(), Counter())
    active = deque(
        (
            order,
            scan_first_leg_durations(
                seed, order, deposits, min_days, deadline, transfer_dv,
                probes_by_duration, depth_stats,
            ),
        )
        for order in orders
    )
    best_state, best_score = incumbent, max(target, official_score)
    solves = 0
    validation_attempts = 0
    rejected = 0
    while active and solves < max_lambert_solves:
        order, scan = active.popleft()
        children = next(scan, None)
        if children is None:
            continue
        misses = transfer_dv.cache_info().misses
        scans_by_order[order] += misses - solves
        solves = misses
        active.append((order, scan))

        for child in children:
            # Allow for the validator's internal/official score agreement tolerance.
            if min(child.collected_materials) + SCORE_TOLERANCE <= best_score:
                depth_stats[-1]["score_skips"] += 1
                continue
            validation_attempts += 1
            try:
                score = validate_route(child, require_full_route=True)
            except ValueError:
                rejected += 1
                continue
            if score > best_score:
                best_state, best_score = child, score

    cache_stats = transfer_dv.cache_info()
    print(f"Transfer cache hits: {cache_stats.hits}")
    print(f"Transfer cache misses: {cache_stats.misses}")
    print(f"Actual search Lambert solves: {solves}/{max_lambert_solves}")
    print("Lambert budget exhausted." if solves >= max_lambert_solves else "Search space exhausted.")
    material_names = ("Gold", "Platinum", "Nickel")
    for order, count in scans_by_order.items():
        print(" -> ".join(material_names[ore] for ore in order), count)
    print("Transfer probes by first-leg duration (including cache hits):")
    for days, count in probes_by_duration.items():
        print(f"  {days} days: {count}")
    print("Ore-stop depth diagnostics (probes include cache hits):")
    for depth, stats in enumerate(depth_stats, start=1):
        print(
            f"  Depth {depth} | probes {stats['probes']} | "
            f"feasible children {stats['feasible_children']} | "
            f"Lambert failures {stats['lambert_failures']} | "
            f"fuel failures {stats['fuel_failures']} | "
            f"time cutoffs {stats['time_cutoffs']} | "
            f"full stays skipped by time {stats['full_stay_time_skips']} | "
            f"visited skips {stats['visited_skips']} | "
            f"transition rejections {stats['transition_rejections']} | "
            f"score skips {stats['score_skips']}"
        )
    print(f"Official candidate validation attempts: {validation_attempts}")
    print(f"Candidate routes rejected during official validation: {rejected}")
    if best_state is incumbent:
        print(f"Saved official incumbent retained: {official_score}")
    else:
        print(f"New official three-stop winner: {best_score}")
        print("Tail stops:", best_state.route[len(seed.route):])
        print("Collected Gold, Platinum, Nickel:", best_state.collected_materials)
    return best_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=2000,
        help="Exclusive asteroid ID limit for Gold and Nickel.",
    )
    parser.add_argument("--max-lambert-solves", type=int, default=320_000)
    args = parser.parse_args()
    run_three_stop_tail_search(args.limit, args.max_lambert_solves)
