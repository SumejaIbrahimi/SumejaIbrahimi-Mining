"""Bounded search over up to four or five replacements for the saved tail."""

import argparse
from functools import lru_cache
from itertools import chain
import json
import math
import time


SAVED_ROUTE = "three_stop_best_result.json"
SCORE_TOLERANCE = 1e-8


def can_still_improve(
    state, target, remaining_stops, ore_deposits, deadline, min_flight, mining_rate_days
):
    """Test relaxed resource bounds, ignoring fuel, geometry and per-stop timing."""
    deficits = [
        max(0.0, target - collected - SCORE_TOLERANCE)
        for collected in state.collected_materials
    ]
    minimum_stops = 0
    for deficit, deposits in zip(deficits, ore_deposits):
        if deficit == 0:
            continue
        available = 0.0
        # Each material list is sorted by descending mass.
        for asteroid_id, mass in deposits:
            if asteroid_id in state.visited_asteroids:
                continue
            if minimum_stops == remaining_stops:
                return False
            minimum_stops += 1
            available += mass
            if available >= deficit:
                break
        else:
            return False

    minimum_days = math.fsum(deficits) * mining_rate_days + minimum_stops * min_flight
    return minimum_days <= deadline - state.current_day + SCORE_TOLERANCE


def run_extended_tail_search(replace_stops=4, max_nodes=1000, max_seconds=15.0):
    """Return the saved incumbent or a strictly better ESA-validated route."""
    if replace_stops not in (4, 5):
        raise ValueError("Replace four or five stops.")
    if max_nodes < 1 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("The node and runtime limits must be positive and finite.")

    # Keep the pruning function testable without PyKEP or the catalog.
    from beam_search import apply_candidate, calculate_transfer_dv
    from beam_search import load_and_replay_baseline, validate_route
    from find_reachable import esa, udp
    from mission import calculate_mining_days
    from three_stop_tail_search import TRANSFER_CACHE_SIZE, TRANSFER_DAYS

    with open(SAVED_ROUTE, encoding="utf-8") as file:
        recorded_score = float(json.load(file)["official_score"])
    states = load_and_replay_baseline(SAVED_ROUTE)
    if len(states) <= replace_stops:
        raise ValueError("The saved route is too short for the requested replacement.")
    seed, incumbent = states[-replace_stops - 1], states[-1]
    best_score = validate_route(incumbent, require_full_route=True)
    if not math.isclose(best_score, recorded_score, rel_tol=0, abs_tol=SCORE_TOLERANCE):
        raise ValueError("The saved route does not match its recorded official score.")

    best_state = incumbent
    deadline = udp.mission_window[1]
    mining_rate_days = esa["TIME_TO_MINE_FULLY"]
    materials = [[], [], [], []]
    for asteroid_id, (material, mass) in enumerate(
        zip(udp.asteroid_materials_types, udp.asteroid_masses)
    ):
        if asteroid_id not in seed.visited_asteroids:
            materials[int(material)].append((asteroid_id, float(mass)))
    for deposits in materials:
        deposits.sort(key=lambda deposit: (-deposit[1], deposit[0]))
    ore_deposits = materials[:3]

    # Same exact key, solver, and bounded cache policy as the three-stop search.
    @lru_cache(maxsize=TRANSFER_CACHE_SIZE)
    def transfer_dv(source_id, departure_day, target_id, arrival_day):
        try:
            dv = calculate_transfer_dv(
                start_id=source_id, target_id=target_id,
                departure_day=departure_day, arrival_day=arrival_day,
            )
        except (RuntimeError, ValueError):
            return math.inf
        return dv if math.isfinite(dv) and dv >= 0 else math.inf

    stop_at = time.perf_counter() + max_seconds
    nodes = probes = pruned = 0

    def visit(state, remaining_stops):
        nonlocal nodes, probes, pruned, best_score, best_state
        if nodes >= max_nodes or time.perf_counter() >= stop_at:
            return
        nodes += 1
        threshold = max(best_score, recorded_score + SCORE_TOLERANCE)
        if min(state.collected_materials) > threshold - SCORE_TOLERANCE:
            try:
                official_score = validate_route(state, require_full_route=True)
            except ValueError:
                pass
            else:
                if official_score > threshold:
                    best_score, best_state = official_score, state

        if not remaining_stops:
            return
        if not can_still_improve(
            state, best_score, remaining_stops, ore_deposits,
            deadline, min(TRANSFER_DAYS), mining_rate_days,
        ):
            pruned += 1
            return

        # Consider scarce deficit ores first. All four types remain available.
        material_order = sorted(
            range(4),
            key=lambda material: (
                material == 3 or state.collected_materials[material] >= best_score,
                len(materials[material]),
            ),
        )
        for material in material_order:
            for asteroid_id, mass in materials[material]:
                if asteroid_id in state.visited_asteroids:
                    continue
                for days in TRANSFER_DAYS:
                    if nodes >= max_nodes or probes >= max_nodes or time.perf_counter() >= stop_at:
                        return
                    arrival_day = state.current_day + days
                    latest_stay = min(60, math.floor(deadline - arrival_day))
                    if latest_stay < 1:
                        break
                    probes += 1
                    dv = transfer_dv(state.current_id, state.current_day, asteroid_id, arrival_day)
                    if dv > state.remaining_dv:
                        continue
                    candidate = {
                        "id": asteroid_id, "mass": mass, "material": material,
                        "arrival_day": arrival_day, "dv": dv,
                    }
                    full_stay = calculate_mining_days(candidate)
                    preferred_stay = min(full_stay, latest_stay)
                    # Try extraction first, then all other legal integer stays.
                    stays = chain(range(preferred_stay, latest_stay + 1), range(1, preferred_stay))
                    for mining_days in stays:
                        if nodes >= max_nodes or time.perf_counter() >= stop_at:
                            return
                        child = apply_candidate(state, candidate, mining_days)
                        if child is not None:
                            visit(child, remaining_stops - 1)

    visit(seed, replace_stops)
    cache_stats = transfer_dv.cache_info()
    print(f"Seed: asteroid {seed.current_id}, ready on day {seed.current_day}")
    print(f"Nodes: {nodes}/{max_nodes}; probes: {probes}/{max_nodes}; bound prunes: {pruned}")
    print(f"Search Lambert solves: {cache_stats.misses}; cache hits: {cache_stats.hits}")
    if best_state is incumbent:
        print(f"Saved official incumbent retained: {best_score}")
    else:
        print(f"New official extended-tail winner: {best_score}")
        print("Replacement stops:", best_state.route[len(seed.route):])
    return best_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace-stops", type=int, choices=(4, 5), default=4)
    parser.add_argument("--max-nodes", type=int, default=1000, help="Cap on states and on probes.")
    parser.add_argument("--max-seconds", type=float, default=15.0, help="Search time budget.")
    args = parser.parse_args()
    run_extended_tail_search(args.replace_stops, args.max_nodes, args.max_seconds)
