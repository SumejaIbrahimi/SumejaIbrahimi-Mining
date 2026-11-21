"""Screen four fixed replacement timings and report the best official result."""

from dataclasses import replace
import subprocess
import sys
import time

from middle_route_experiment import SOURCE, SUCCESS_SCORE, TARGET_ID, replace_stop


TRIALS = ((2985, 1490), (2985, 1500), (9419, 1490), (9419, 1500))
TIME_BUDGET_SECONDS = 15


def trial_route(route, index, asteroid_id, arrival_day):
    previous = route[index - 1]
    if previous.asteroid_id != 375 or previous.arrival_day + previous.mining_days != 1415:
        raise ValueError("The saved departure from asteroid 375 has changed.")
    replacement = replace_stop(route, asteroid_id)
    return (
        replacement[:index]
        + (replace(replacement[index], arrival_day=arrival_day),)
        + replacement[index + 1:]
    )


def run():
    from beam_search import (
        apply_candidate,
        calculate_transfer_dv,
        esa,
        load_and_replay_baseline,
        validate_route,
    )

    started = time.perf_counter()
    saved = load_and_replay_baseline(SOURCE)
    original = saved[-1].route
    index = next(i for i, stop in enumerate(original) if stop.asteroid_id == TARGET_ID)
    prefix = saved[index - 1]
    catalog = esa["data"]

    def transfer(source, destination):
        return calculate_transfer_dv(
            source.asteroid_id, destination.asteroid_id,
            source.arrival_day + source.mining_days, destination.arrival_day,
        )

    suffix_dv = [
        transfer(original[i - 1], original[i])
        for i in range(index + 2, len(original))
    ]
    best = None
    for asteroid_id, arrival_day in TRIALS:
        route = trial_route(original, index, asteroid_id, arrival_day)
        row = catalog[asteroid_id]
        if asteroid_id in saved[-1].visited_asteroids or row[-1] != 1:
            raise ValueError(f"Asteroid {asteroid_id} is visited or is not material 1.")
        label = f"Candidate {asteroid_id} on day {arrival_day}"
        try:
            inbound = transfer(route[index - 1], route[index])
            outbound = transfer(route[index], route[index + 1])
        except (RuntimeError, ValueError) as error:
            print(f"{label} | transfer failed ({error})")
            continue
        label += f" | inbound {inbound:.2f} m/s | outbound {outbound:.2f} m/s"

        state = prefix
        for stop, dv in zip(route[index:], [inbound, outbound, *suffix_dv]):
            deposit = catalog[stop.asteroid_id]
            state = apply_candidate(state, {
                "id": stop.asteroid_id,
                "arrival_day": stop.arrival_day,
                "dv": dv,
                "mass": float(deposit[-2]),
                "material": int(deposit[-1]),
            }, stop.mining_days)
            if state is None:
                print(f"{label} | infeasible at asteroid {stop.asteroid_id}")
                break
        else:
            print(f"{label} | fuel feasible | {state.remaining_dv:.2f} m/s at finish")
            if best is None or (min(state.collected_materials), state.remaining_dv) > (
                min(best.collected_materials), best.remaining_dv,
            ):
                best = state

    if best is None:
        print("No fuel-feasible complete route. Success target not met.")
        return
    try:
        score = validate_route(best, require_full_route=True)
    except (RuntimeError, ValueError) as error:
        print(f"Full-route validation failed ({error}). Success target not met.")
        return
    stop = best.route[index]
    print(f"Best trial: asteroid {stop.asteroid_id} on day {stop.arrival_day} "
          f"| official score {score!r} | {time.perf_counter() - started:.3f} s "
          f"| success target {'met' if score >= SUCCESS_SCORE else 'not met'}")


def main():
    # A child process lets the parent enforce the deadline during native solves.
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-u", "-c",
             "from middle_route_timing_experiment import run; run()"],
            timeout=TIME_BUDGET_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print("15-second budget reached. No completed official result was retained.")
        return 1
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
