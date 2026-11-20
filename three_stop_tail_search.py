"""Search exactly three ore stops after the saved route's stop at asteroid 332."""

import argparse
from collections import deque
from itertools import permutations
import math

from beam_search import apply_candidate, calculate_transfer_dv, load_and_replay_baseline
from beam_search import validate_route
from find_reachable import esa, udp
from mission import calculate_mining_days


SAVED_SCORE = 3.846180920595038
SCORE_TOLERANCE = 1e-8
TRANSFER_DAYS = (20, 40, 60, 80, 100)


def minimum_mining_days(amount, target, mining_rate_days):
    days = max(1, math.floor((target - amount) * mining_rate_days))
    while amount + days / mining_rate_days <= target:
        days += 1
    return days


def scan_order(state, order, deposits, min_days, deadline):
    """Yield once per Lambert solve, with any complete routes from that solve."""
    material, *remaining_order = order
    minimum_stay = min_days[material]
    remaining_time = sum(min_days[ore] + TRANSFER_DAYS[0] for ore in remaining_order)

    for flight_days in TRANSFER_DAYS:
        arrival_day = state.current_day + flight_days
        latest_stay = math.floor(deadline - arrival_day - remaining_time)
        if minimum_stay > latest_stay:
            break

        for asteroid_id, mass, full_stay in deposits[material]:
            if asteroid_id in state.visited_asteroids:
                continue

            dv = calculate_transfer_dv(
                state.current_id, asteroid_id, state.current_day, arrival_day
            )
            if not math.isfinite(dv) or not 0 <= dv <= state.remaining_dv:
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
            children = []
            for mining_days in stays:
                child = apply_candidate(state, candidate, mining_days)
                if child is not None:
                    children.append(child)

            if remaining_order:
                yield ()
                for child in children:
                    yield from scan_order(
                        child, remaining_order, deposits, min_days, deadline
                    )
            else:
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
    for asteroid_id in range(min(limit, len(udp.asteroids))):
        material = int(udp.asteroid_materials_types[asteroid_id])
        if material == 3 or asteroid_id in seed.visited_asteroids:
            continue
        mass = float(udp.asteroid_masses[asteroid_id])
        if seed.collected_materials[material] + mass <= target:
            continue
        full_stay = calculate_mining_days({"mass": mass}, partial_mining=False)
        deposits[material].append((asteroid_id, mass, full_stay))

    for ore_deposits in deposits:
        ore_deposits.sort(key=lambda deposit: (-deposit[1], deposit[0]))

    # Interleave orders by solve so no single order consumes the shared budget.
    orders = list(permutations(range(3))) if all(deposits) else []
    scans_by_order = dict.fromkeys(orders, 0)
    active = deque(
        (order, scan_order(seed, order, deposits, min_days, deadline))
        for order in orders
    )
    best_state, best_score = incumbent, max(target, official_score)
    solves = 0
    rejected = 0
    while active and solves < max_lambert_solves:
        order, scan = active.popleft()
        children = next(scan, None)
        if children is None:
            continue
        solves += 1
        scans_by_order[order] += 1
        active.append((order, scan))

        for child in children:
            # Allow for the validator's internal/official score agreement tolerance.
            if min(child.collected_materials) + SCORE_TOLERANCE <= best_score:
                continue
            try:
                score = validate_route(child, require_full_route=True)
            except ValueError:
                rejected += 1
                continue
            if score > best_score:
                best_state, best_score = child, score

    print(f"Search Lambert solves: {solves}/{max_lambert_solves}")
    print("Search stopped at the budget." if active else "Search space exhausted.")
    material_names = ("Gold", "Platinum", "Nickel")
    for order, count in scans_by_order.items():
        print(" -> ".join(material_names[ore] for ore in order), count)
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
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--max-lambert-solves", type=int, default=320_000)
    args = parser.parse_args()
    run_three_stop_tail_search(args.limit, args.max_lambert_solves)
