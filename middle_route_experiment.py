"""Try a fixed-schedule replacement for asteroid 831 in the saved route."""

from dataclasses import replace
import json
import math
from pathlib import Path
import time


SOURCE = Path("three_stop_best_result.json")
INCUMBENT_SCORE = 3.8631533739384336
SUCCESS_SCORE = 3.8650
MIN_DEPOSIT = 0.91125
CANDIDATE_LIMIT = 20
TARGET_ID = 831


def select_candidates(catalog, visited_ids):
    """Rank eligible IDs by normalized semimajor axis, eccentricity and inclination."""
    visited = set(visited_ids)
    _, axis, eccentricity, inclination, *_ = catalog[TARGET_ID]

    def proximity(row):
        asteroid_id, candidate_axis, candidate_eccentricity, candidate_inclination, *_ = row
        return (
            math.hypot(
                (candidate_axis - axis) / axis,
                candidate_eccentricity - eccentricity,
                candidate_inclination - inclination,
            ),
            int(asteroid_id),
        )

    eligible = [
        row for row in catalog
        if int(row[0]) not in visited and row[-1] == 1 and row[-2] >= MIN_DEPOSIT
    ]
    return [int(row[0]) for row in sorted(eligible, key=proximity)[:CANDIDATE_LIMIT]]


def replace_stop(route, candidate_id):
    """Return a route with only the target asteroid ID changed."""
    index = next(i for i, stop in enumerate(route) if stop.asteroid_id == TARGET_ID)
    target, following = route[index:index + 2]
    if (target.arrival_day, target.mining_days, following.asteroid_id,
            following.arrival_day) != (1495, 28, 332, 1583):
        raise ValueError("The saved schedule around asteroid 831 has changed.")
    replacement = replace(target, asteroid_id=candidate_id)
    return route[:index] + (replacement,) + route[index + 1:]


def main():
    from beam_search import (
        apply_candidate,
        calculate_transfer_dv,
        esa,
        load_and_replay_baseline,
        validate_route,
    )

    started = time.perf_counter()
    saved_states = load_and_replay_baseline(SOURCE)
    saved_route = saved_states[-1].route
    index = next(i for i, stop in enumerate(saved_route) if stop.asteroid_id == TARGET_ID)
    catalog = esa["data"]
    candidates = select_candidates(catalog, saved_states[-1].visited_asteroids)

    def transfer(source, destination):
        return calculate_transfer_dv(
            source.asteroid_id, destination.asteroid_id,
            source.arrival_day + source.mining_days, destination.arrival_day,
        )

    # The suffix after 332 has the same asteroids and dates for every candidate.
    suffix_dv = [
        transfer(saved_route[i - 1], saved_route[i])
        for i in range(index + 2, len(saved_route))
    ]
    winner = None
    best_score = INCUMBENT_SCORE
    for candidate_id in candidates:
        candidate_started = time.perf_counter()
        route = replace_stop(saved_route, candidate_id)
        changed_dv = [None, None]
        official_score = None
        try:
            for leg, i in enumerate((index, index + 1)):
                changed_dv[leg] = transfer(route[i - 1], route[i])
            state = saved_states[index - 1]
            for stop, dv in zip(route[index:], changed_dv + suffix_dv):
                row = catalog[stop.asteroid_id]
                state = apply_candidate(state, {
                    "id": stop.asteroid_id,
                    "arrival_day": stop.arrival_day,
                    "dv": dv,
                    "mass": float(row[-2]),
                    "material": int(row[-1]),
                }, stop.mining_days)
                if state is None:
                    raise ValueError(f"Route infeasible at asteroid {stop.asteroid_id}")
            official_score = validate_route(state, require_full_route=True)
            if official_score > best_score:
                winner, best_score = state, official_score
            status = "valid"
        except (RuntimeError, ValueError) as error:
            status = str(error)
        inbound, outbound = ("n/a" if dv is None else f"{dv:.2f}" for dv in changed_dv)
        score_text = "n/a" if official_score is None else repr(official_score)
        print(f"Candidate {candidate_id} | inbound {inbound} m/s | outbound {outbound} m/s"
              f" | official score {score_text} | {time.perf_counter() - candidate_started:.3f} s"
              f" | {status}")

    runtime = time.perf_counter() - started
    print(f"Tested {len(candidates)} candidates in {runtime:.3f} s. "
          f"Best official score including incumbent: {best_score!r}. "
          f"Success target {'met' if best_score >= SUCCESS_SCORE else 'not met'}.")
    if winner is not None:
        output = Path(f"middle_route_result_{time.time_ns()}.json")
        result = {
            "algorithm": "Fixed-schedule middle-route replacement",
            "source": SOURCE.name,
            "replaced_asteroid": TARGET_ID,
            "replacement_asteroid": winner.route[index].asteroid_id,
            "previous_official_score": INCUMBENT_SCORE,
            "score": best_score,
            "official_score": best_score,
            "visited_asteroids": [stop.asteroid_id for stop in winner.route],
            "arrival_times": [stop.arrival_day for stop in winner.route],
            "mining_times": [stop.mining_days for stop in winner.route],
            "collected_materials": dict(enumerate(winner.collected_materials)),
            "remaining_dv": winner.remaining_dv,
            "final_day": winner.current_day,
            "runtime_seconds": runtime,
        }
        with output.open("x", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
