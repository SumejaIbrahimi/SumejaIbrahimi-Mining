"""Search the six stops after the incumbent departs asteroid 5420."""

from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from anchored_two_stop_search import (
    BudgetExceeded, INCUMBENT_SCORE, LEG_COUNT, SOLVE_LIMIT, TIME_LIMIT,
    TOLERANCE, TransferCache, route_rank, shortlist,
)
from early_gold_experiment import export_result


SOURCE = Path("middle_gold_best_result.json")
OUTPUT = Path("pareto_tail_best_result.json")
MISSION_END = 1827
TAIL_START = 17
BEAM_WIDTH = 16
ARRIVAL_OFFSETS = (-12, -8, -4, 0, 4, 8, 12)
STAY_OFFSETS = (-4, -2, 0, 2, 5, 9)
STAGE_LIMITS = (3, 8, 3, 1, 3, 1)
TAIL_IDS = (831, 4088, 8719, 1266, 735, 7735)


def context(state):
    return state.current_id, state.current_day, state.visited_asteroids


def resources(state):
    return (state.remaining_dv, *state.collected_materials)


def pareto_front(states):
    fronts = defaultdict(list)
    for state in states:
        front = fronts[context(state)]
        values = resources(state)
        if any(all(a >= b for a, b in zip(resources(other), values)) for other in front):
            continue  # Also coalesce equal resource vectors in the same context.
        front[:] = [
            other for other in front
            if not all(a >= b for a, b in zip(values, resources(other)))
        ]
        front.append(state)
    return [state for front in fronts.values() for state in front]


def choose_beam(states, future_yields):
    def priority(state):
        projected = min(total + extra for total, extra in zip(state.collected_materials, future_yields))
        return projected, state.remaining_dv, -state.current_day

    ranked = sorted(pareto_front(states), key=priority, reverse=True)
    selected = []
    deferred = []
    timings = set()
    for state in ranked:
        timing = state.current_id, state.current_day
        if timing in timings:
            deferred.append(state)
            continue
        selected.append(state)
        timings.add(timing)
        if len(selected) == BEAM_WIDTH:
            return selected
    return selected + deferred[:BEAM_WIDTH - len(selected)]


def mining_stays(nominal, mass, arrival, terminal):
    latest = min(60, MISSION_END - arrival)
    options = {nominal + offset for offset in STAY_OFFSETS}
    options.add(math.ceil(mass * 30))
    if terminal:
        options.add(latest)
    return sorted(stay for stay in options if 1 <= stay <= latest)


def search_tail(control, options, nominal_remaining, mass_for, stop_type, advance, check_time):
    """Expand bounded frontiers and retain the incumbent path outside beam pruning."""
    beam = []
    best = control[-1]
    for stage, ids in enumerate(options):
        nominal = control[stage + 1].route[-1]
        terminal = stage == len(options) - 1
        children = []
        parents = [control[stage], *(state for state in beam if state != control[stage])]
        for parent in parents:
            for asteroid_id in ids:
                if asteroid_id in parent.visited_asteroids:
                    continue
                mass = mass_for(asteroid_id)
                for offset in ARRIVAL_OFFSETS:
                    check_time()
                    arrival = nominal.arrival_day + offset
                    if arrival - parent.current_day < 0.1:
                        continue
                    for stay in mining_stays(nominal.mining_days, mass, arrival, terminal):
                        child = advance(parent, stop_type(asteroid_id, arrival, stay))
                        if child is None:
                            # Stay changes yield and departure, while the incoming cost is fixed.
                            break
                        if terminal:
                            if route_rank(child) > route_rank(best):
                                best = child
                        else:
                            children.append(child)
        if not terminal:
            beam = choose_beam(children, nominal_remaining[stage])
    return best


def run(deadline):
    from beam_search import (
        BeamState, RouteStop, apply_candidate, calculate_transfer_dv, esa, udp, validate_route,
    )

    saved = json.loads(SOURCE.read_text(encoding="utf-8"))
    route = tuple(RouteStop(*fields) for fields in zip(
        saved["visited_asteroids"], saved["arrival_times"], saved["mining_times"], strict=True,
    ))
    if (len(route) != LEG_COUNT + 1 or saved["official_score"] != INCUMBENT_SCORE
            or route[TAIL_START - 1] != RouteStop(5420, 1455, 30)
            or tuple(stop.asteroid_id for stop in route[TAIL_START:]) != TAIL_IDS):
        raise ValueError("Saved incumbent does not match the expected tail.")

    transfers = TransferCache(calculate_transfer_dv, deadline)

    def mass_for(asteroid_id):
        return float(udp.asteroid_masses[asteroid_id])

    def advance(state, stop):
        asteroid_id = stop.asteroid_id
        dv = transfers.transfer(state.current_id, state.current_day, asteroid_id, stop.arrival_day)
        return apply_candidate(state, {
            "id": asteroid_id, "arrival_day": stop.arrival_day, "dv": dv,
            "material": int(udp.asteroid_materials_types[asteroid_id]),
            "mass": mass_for(asteroid_id),
        }, stop.mining_days)

    first = route[0]
    replay = [BeamState(first.asteroid_id, first.arrival_day + first.mining_days, 10000.0,
                        (0.0, 0.0, 0.0), frozenset({first.asteroid_id}), (first,))]
    for stop in route[1:]:
        if stop.arrival_day - replay[-1].current_day < 0.1:
            raise ValueError(f"Invalid incumbent transfer into asteroid {stop.asteroid_id}.")
        child = advance(replay[-1], stop)
        if child is None:
            raise ValueError(f"Incumbent replay failed at asteroid {stop.asteroid_id}.")
        replay.append(child)
    incumbent = replay[-1]
    if not math.isclose(route_rank(incumbent)[0], INCUMBENT_SCORE, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError("Incumbent replay differs from the saved official score.")

    epoch = esa["T_START"].mjd2000
    options = []
    for index, limit in enumerate(STAGE_LIMITS, start=TAIL_START):
        transfers.check_time()
        stop = route[index]
        asteroid_id = stop.asteroid_id
        if limit == 1:
            options.append([asteroid_id])
            continue
        previous, following = route[index - 1], route[index + 1]
        incoming = stop.arrival_day - previous.arrival_day - previous.mining_days
        outgoing = following.arrival_day - stop.arrival_day - stop.mining_days
        material = int(udp.asteroid_materials_types[asteroid_id])
        # A gold replacement must add gold. Allow platinum to trade some yield for time or fuel.
        minimum = min(mass_for(asteroid_id), stop.mining_days / 30)
        if material == 1:
            minimum = max(0.0, minimum - 0.2)
        options.append(shortlist(
            udp, incumbent.visited_asteroids, epoch, material, asteroid_id,
            ((previous.asteroid_id, incoming), (following.asteroid_id, outgoing)),
            tuple(stop.arrival_day + offset for offset in ARRIVAL_OFFSETS),
            max(stop.mining_days + max(STAY_OFFSETS), math.ceil(mass_for(asteroid_id) * 30)),
            minimum, limit,
        ))

    nominal_remaining = [
        tuple(final - spent for final, spent in zip(incumbent.collected_materials, state.collected_materials))
        for state in replay[TAIL_START:]
    ]
    print(f"Tail shortlists: {options}")
    best = search_tail(replay[TAIL_START - 1:], options, nominal_remaining, mass_for,
                       RouteStop, advance, transfers.check_time)
    print(f"Replay and screening: {transfers.solves} Lambert solves (limit {SOLVE_LIMIT} with validation).")
    if route_rank(best)[0] <= INCUMBENT_SCORE:
        print("No improving feasible tail found. No official validation used.")
        return

    print(f"Best screened score: {route_rank(best)[0]:.15f} | tail {best.route[TAIL_START:]}")
    transfers.check_time()
    score = validate_route(best, require_full_route=True)
    transfers.check_time()
    print(f"Official ESA score {score!r} | improvement {score - INCUMBENT_SCORE:+.12f}")
    if score > INCUMBENT_SCORE:
        try:
            export_result(best, score, output=OUTPUT, algorithm="Pareto tail Beam Search")
        except FileExistsError:
            print(f"{OUTPUT} already exists. Kept the existing result.")
        else:
            print(f"Saved {OUTPUT}")


def main():
    deadline = time.monotonic() + TIME_LIMIT
    try:
        worker = subprocess.run(
            [sys.executable, "-B", "-u", "-c",
             "import sys; from pareto_tail_search import run; run(float(sys.argv[1]))", str(deadline)],
            timeout=max(0.0, deadline - time.monotonic()),
        )
    except subprocess.TimeoutExpired:
        print(f"{TIME_LIMIT}-second limit reached. Worker stopped.")
        return 1
    return worker.returncode


if __name__ == "__main__":
    sys.exit(main())
