import numpy as np
import pykep as pk
from dataclasses import dataclass
import math
import json
from pathlib import Path
import time
from contextlib import redirect_stdout
from io import StringIO

from mission import calculate_mining_days
from find_reachable import search_multiple_transfers, esa, udp


@dataclass(frozen=True)
class RouteStop:
    asteroid_id: int
    arrival_day: float
    mining_days: int


@dataclass(frozen=True)
class BeamState:
    current_id: int
    current_day: float
    remaining_dv: float
    collected_materials: tuple
    visited_asteroids: frozenset
    route: tuple

def calculate_transfer_dv(start_id, target_id,
                          departure_day, arrival_day):

    r1, v1 = udp.asteroids[start_id].eph(
        esa["T_START"].mjd2000 + departure_day
    )

    r2, v2 = udp.asteroids[target_id].eph(
        esa["T_START"].mjd2000 + arrival_day
    )

    trajectory = pk.lambert_problem(
        r1=r1,
        r2=r2,
        tof=(arrival_day - departure_day) * pk.DAY2SEC,
        mu=udp.MU,
        cw=False,
        max_revs=0
    )

    dv1 = np.linalg.norm(
        np.array(v1) - np.array(trajectory.get_v1()[0])
    )

    dv2 = np.linalg.norm(
        np.array(v2) - np.array(trajectory.get_v2()[0])
    )

    return float(dv1 + dv2)

def load_and_replay_baseline(
    filename="baseline_result.json"
):
    with open(filename, "r") as file:
        baseline = json.load(file)

    ids = baseline["visited_asteroids"]
    arrivals = baseline["arrival_times"]
    mining_times = baseline["mining_times"]

    state = BeamState(
        current_id=ids[0],
        current_day=arrivals[0] + mining_times[0],
        remaining_dv=10000.0,
        collected_materials=(0.0, 0.0, 0.0),
        visited_asteroids=frozenset([ids[0]]),
        route=(
            RouteStop(
                ids[0],
                arrivals[0],
                mining_times[0]
            ),
        )
    )

    states = [state]

    for i in range(1, len(ids)):

        asteroid_id = ids[i]
        arrival_day = arrivals[i]
        mining_days = mining_times[i]

        dv = calculate_transfer_dv(
            state.current_id,
            asteroid_id,
            state.current_day,
            arrival_day
        )

        remaining_dv = state.remaining_dv - dv

        material = int(
            udp.asteroid_materials_types[asteroid_id]
        )

        mass = float(
            udp.asteroid_masses[asteroid_id]
        )

        mined_mass = min(mass, mining_days / 30.0)

        materials = list(state.collected_materials)

        if material == 3:
            remaining_dv = min(
                10000.0,
                remaining_dv + mined_mass * 10000
            )
        else:
            materials[material] += mined_mass

        state = BeamState(
            current_id=asteroid_id,
            current_day=arrival_day + mining_days,
            remaining_dv=remaining_dv,
            collected_materials=tuple(materials),
            visited_asteroids=(
                state.visited_asteroids | {asteroid_id}
            ),
            route=state.route + (
                RouteStop(
                    asteroid_id,
                    arrival_day,
                    mining_days
                ),
            )
        )

        states.append(state)

    return states

def validate_route(state, verbose=False, *, require_full_route=False):

    route = state.route

    if not route:
        raise ValueError("Itinerari eshte i zbrazet.")

    ids = [stop.asteroid_id for stop in route]
    arrivals = [stop.arrival_day for stop in route]
    mining_times = [stop.mining_days for stop in route]

    if len(ids) != len(set(ids)):
        raise ValueError("Ka asteroide te perseritur.")

    if arrivals[0] != 0 or mining_times[0] != 1:
        raise ValueError("Gjendja fillestare eshte e pavlefshme.")

    remaining_dv = 10000.0
    materials = [0.0, 0.0, 0.0]

    for i in range(len(route)):

        if not 0 <= ids[i] < len(udp.asteroids):
            raise ValueError("ID e pavlefshme.")

        if not math.isfinite(arrivals[i]):
            raise ValueError("Kohe e pavlefshme.")

        if not 1 <= mining_times[i] <= 60:
            raise ValueError("Kohezgjatje e pavlefshme e minimit.")

        if arrivals[i] + mining_times[i] > 1827:
            raise ValueError("Afati i misionit eshte tejkaluar.")

        if i == 0:
            continue

        departure_day = arrivals[i - 1] + mining_times[i - 1]

        if arrivals[i] - departure_day < 0.1:
            raise ValueError("Koha e udhetimit eshte e pavlefshme.")

        dv = calculate_transfer_dv(
            ids[i - 1],
            ids[i],
            departure_day,
            arrivals[i]
        )

        if not math.isfinite(dv) or dv > remaining_dv:
            raise ValueError("Delta-v i pamjaftueshem.")

        remaining_dv -= dv

        material = int(
            udp.asteroid_materials_types[ids[i]]
        )

        mass = float(
            udp.asteroid_masses[ids[i]]
        )

        mined_mass = min(mass, mining_times[i] / 30.0)

        if material == 3:
            remaining_dv = min(
                10000.0,
                remaining_dv + mined_mass * 10000
            )
        else:
            materials[material] += mined_mass

    if not np.allclose(
        materials,
        state.collected_materials,
        rtol=0,
        atol=1e-8
    ):
        raise ValueError("Materialet nuk perputhen.")

    if not math.isclose(
        remaining_dv,
        state.remaining_dv,
        rel_tol=0,
        abs_tol=1e-6
    ):
        raise ValueError("Delta-v nuk perputhet.")

    if state.current_id != ids[-1]:
        raise ValueError("Asteroidi aktual nuk perputhet.")

    if state.current_day != arrivals[-1] + mining_times[-1]:
        raise ValueError("Dita finale nuk perputhet.")

    if state.visited_asteroids != frozenset(ids):
        raise ValueError("Asteroidet e vizituar nuk perputhen.")

    solution = arrivals + mining_times + ids

    chromosome = esa["convert_to_chromosome"](solution)

    if require_full_route:
        trace = StringIO()
        with redirect_stdout(trace):
            fitness = udp.fitness(chromosome, verbose=True)
        output = trace.getvalue()
        credited_ids = [
            int(line.split("\t", 1)[0])
            for line in output.splitlines()
            if "\t" in line and line.split("\t", 1)[0].isdigit()
        ]
        if credited_ids != ids[1:]:
            raise ValueError("The ESA evaluator did not credit the full route.")
        if verbose:
            print(output, end="")
    else:
        fitness = udp.fitness(chromosome, verbose=verbose)

    official_score = float(fitness[0])
    internal_score = min(materials)

    if not math.isclose(
        official_score,
        internal_score,
        rel_tol=0,
        abs_tol=1e-8
    ):
        raise ValueError("Score zyrtar nuk perputhet.")

    return official_score

def apply_candidate(state, candidate, mining_days):

    asteroid_id = candidate["id"]
    arrival_day = candidate["arrival_day"]
    dv = candidate["dv"]

    if asteroid_id in state.visited_asteroids:
        return None

    if not math.isfinite(dv) or dv < 0:
        return None

    if dv > state.remaining_dv:
        return None

    if not math.isfinite(arrival_day):
        return None

    if arrival_day - state.current_day < 0.1:
        return None

    if not 1 <= mining_days <= 60:
        return None

    if arrival_day + mining_days > 1827:
        return None

    remaining_dv = state.remaining_dv - dv

    material = candidate["material"]
    mass = candidate["mass"]

    mined_mass = min(mass, mining_days / 30.0)

    materials = list(state.collected_materials)

    if material == 3:
        remaining_dv = min(
            10000.0,
            remaining_dv + mined_mass * 10000
        )
    else:
        materials[material] += mined_mass

    return BeamState(
        current_id=asteroid_id,
        current_day=arrival_day + mining_days,
        remaining_dv=remaining_dv,
        collected_materials=tuple(materials),
        visited_asteroids=(
            state.visited_asteroids | {asteroid_id}
        ),
        route=state.route + (
            RouteStop(
                asteroid_id,
                arrival_day,
                mining_days
            ),
        )
    )

def expand_state(state, limit=2000,
                 transfer_days=None,
                 partial_mining=True, *, deficits=None, on_probes=None):

    candidates = search_multiple_transfers(
        start_id=state.current_id,
        departure_day=state.current_day,
        available_dv=state.remaining_dv,
        visited_asteroids=state.visited_asteroids,
        limit=limit,
        transfer_days=transfer_days,
        deficits=deficits,
        on_probes=on_probes,
    )

    children = []

    for candidate in candidates:

        mining_days = calculate_mining_days(
            candidate,
            partial_mining
        )

        child = apply_candidate(
            state,
            candidate,
            mining_days
        )

        if child is not None:
            children.append(child)

    return children

def rank_state(state, target):
    deficit = sum(
        max(0.0, target - amount)
        for amount in state.collected_materials
    )

    fuel_fraction = state.remaining_dv / 10000.0

    remaining_days = 1827 - state.current_day

    fuel_weight = min(
        1.0,
        max(0.0, remaining_days / 200.0)
    )

    time_fraction = remaining_days / 1827.0

    rank = (
        -deficit
        + fuel_weight * fuel_fraction
        + 0.25 * time_fraction
    )

    return rank

def select_children(children, target, branch_factor=6):

    ranked = sorted(
        children,
        key=lambda child: rank_state(child, target),
        reverse=True
    )

    selected = []
    destination_counts = {}

    # Ruajme diversitetin e materialeve
    for material in range(4):

        for child in ranked:

            if len(selected) >= branch_factor:
                return selected

            child_material = int(
                udp.asteroid_materials_types[child.current_id]
            )

            if child_material == material:

                selected.append(child)

                destination_counts[child.current_id] = (
                    destination_counts.get(child.current_id, 0) + 1
                )

                break

    # Plotesojme vendet e mbetura sipas rank-ut
    for child in ranked:

        if len(selected) >= branch_factor:
            break

        if child in selected:
            continue

        count = destination_counts.get(child.current_id, 0)

        if count >= 2:
            continue

        selected.append(child)

        destination_counts[child.current_id] = count + 1

    return selected

def select_beam(states, target, beam_width=4, *, on_timing_reserve=None):

    ranked = sorted(
        states,
        key=lambda state: rank_state(state, target),
        reverse=True
    )

    selected = []
    used_destinations = set()

    # Fillimisht zgjedhim destinacione te ndryshme
    for state in ranked:

        if len(selected) >= beam_width:
            break

        if state.current_id in used_destinations:
            continue

        selected.append(state)
        used_destinations.add(state.current_id)

    if on_timing_reserve is not None and beam_width > 1 and len(selected) == beam_width:
        normal_pick = selected[-1]
        kept_by_destination = {state.current_id: state for state in selected[:-1]}
        normal_rank = rank_state(normal_pick, target)
        for state in ranked:
            if rank_state(state, target) <= normal_rank:
                break
            kept = kept_by_destination.get(state.current_id)
            if (
                kept is not None
                and state.visited_asteroids == kept.visited_asteroids
                and state.current_day != kept.current_day
            ):
                selected[-1] = state
                on_timing_reserve()
                break

    # Plotesojme vendet e mbetura
    for state in ranked:

        if len(selected) >= beam_width:
            break

        if state not in selected:
            selected.append(state)

    return selected

def run_beam_search(
    start_index=12,
    beam_width=4,
    branch_factor=6,
    limit=2000,
    max_scans=64,
    transfer_days=None,
    partial_mining=True,
    *,
    seed_state=None,
    incumbent_state=None,
    timing_reserve=False,
    candidate_strategy="id-prefix",
):

    if candidate_strategy not in ("id-prefix", "catalog-wide"):
        raise ValueError(f"Unknown candidate strategy: {candidate_strategy}")
    start_time = time.perf_counter()

    if (seed_state is None) != (incumbent_state is None):
        raise ValueError("Supply both seed_state and incumbent_state.")

    custom_seed = seed_state is not None
    if not custom_seed:
        baseline_states = load_and_replay_baseline()
        seed_state = baseline_states[start_index]
        incumbent_state = baseline_states[-1]

    best_state = incumbent_state
    best_score = validate_route(best_state, require_full_route=custom_seed)
    beam = [seed_state]

    scans = 0
    iteration = 0
    timing_reserve_activations = 0
    lambert_probes = 0

    def count_probes(count):
        nonlocal lambert_probes
        lambert_probes += count

    def record_timing_reserve():
        nonlocal timing_reserve_activations
        timing_reserve_activations += 1

    while beam and scans < max_scans:

        iteration += 1

        target = best_score + 0.25

        next_states = []

        for state in beam:

            if scans >= max_scans:
                break

            if state.current_day >= 1827:
                continue

            deficits = None
            if candidate_strategy == "catalog-wide":
                deficits = tuple(max(0.0, target - mass) for mass in state.collected_materials)
            children = expand_state(
                state,
                limit=limit,
                transfer_days=transfer_days,
                partial_mining=partial_mining,
                deficits=deficits,
                on_probes=count_probes,
            )

            scans += 1

            # Kontrollojme score para eliminimit
            for child in children:

                score = min(child.collected_materials)

                if score > best_score + 1e-8:

                    try:
                        official_score = validate_route(
                            child, require_full_route=custom_seed
                        )
                    except ValueError:
                        if not custom_seed:
                            raise
                        continue

                    if official_score > best_score + 1e-8:

                        best_state = child
                        best_score = official_score

                        if not custom_seed:
                            print("Score i ri me i mire:", best_score)

            # Zgjedhim deget qe do te vazhdojne
            continuing_children = [
                child for child in children
                if child.current_day < 1827
            ]
            
            print(
    f"DEBUG | Asteroidi: {state.current_id} | "
    f"Dita: {state.current_day:.1f} | "
    f"DV: {state.remaining_dv:.1f} | "
    f"Kandidate: {len(children)} | "
    f"Per vazhdim: {len(continuing_children)}"
)
            selected = select_children(
                continuing_children,
                target,
                branch_factor
            )

            next_states.extend(selected)

        beam = select_beam(
            next_states,
            target,
            beam_width,
            on_timing_reserve=record_timing_reserve if timing_reserve else None,
        )

        print(
            f"Iteracioni: {iteration} | "
            f"Beam: {len(beam)} | "
            f"Scans: {scans} | "
            f"Best score: {best_score:.6f}"
        )

    elapsed = time.perf_counter() - start_time

    print("\n--- Statistikat e Beam Search ---")
    print("Iteracionet:", iteration)
    print("Skanimet:", scans)
    print(f"Candidate strategy: {candidate_strategy} | Search Lambert probes: {lambert_probes}")
    if timing_reserve:
        print(f"Timing reserve activations: {timing_reserve_activations}/{iteration} iterations")
    print("Koha:", round(elapsed, 2), "sekonda")

    return best_state


CATALOG_WIDE_RESULT = Path("catalog_wide_best_result.json")


def export_catalog_wide_result(state, official_score):
    """Save a fully validated CLI result using the existing route JSON schema."""
    data = {
        "algorithm": "Beam Search",
        "candidate_strategy": "catalog-wide",
        "score": min(state.collected_materials),
        "official_score": official_score,
        "visited_asteroids": [stop.asteroid_id for stop in state.route],
        "arrival_times": [stop.arrival_day for stop in state.route],
        "mining_times": [stop.mining_days for stop in state.route],
        "collected_materials": dict(enumerate(state.collected_materials)),
        "remaining_dv": state.remaining_dv,
        "final_day": state.current_day,
    }
    payload = json.dumps(data, indent=4, allow_nan=False)
    with CATALOG_WIDE_RESULT.open("x", encoding="utf-8") as file:
        file.write(payload + "\n")


def run_tail_search():
    """Search for a validated improvement after the saved stop at asteroid 332."""
    saved_score = 3.846180920595038
    tolerance = 1e-8
    states = load_and_replay_baseline("beam_best_result.json")
    if len(states) <= 17:
        raise ValueError("The saved winner is missing the tail-search seed.")

    seed = states[17]
    incumbent = states[-1]
    if seed.current_id != 332 or seed.current_day != 1606:
        raise ValueError("Expected asteroid 332, ready to depart on day 1606.")

    official_score = validate_route(incumbent, require_full_route=True)
    if not math.isclose(official_score, saved_score, rel_tol=0, abs_tol=tolerance):
        raise ValueError("The saved winner's official score differs from the expected score.")

    winner = run_beam_search(
        beam_width=8,
        branch_factor=6,
        limit=2000,
        max_scans=32,
        partial_mining=True,
        seed_state=seed,
        incumbent_state=incumbent,
        timing_reserve=True,
    )
    if winner is incumbent:
        return incumbent

    if winner.route[:len(seed.route)] != seed.route:
        raise ValueError("The tail-search winner changed the saved prefix.")

    official_score = validate_route(winner, require_full_route=True)
    if official_score <= saved_score + tolerance:
        return incumbent

    print("New official tail-search winner:", official_score)
    return winner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Beam Search with optional catalog-wide export.")
    parser.add_argument("--candidate-strategy", choices=("id-prefix", "catalog-wide"),
                        default="id-prefix")
    parser.add_argument("--incumbent", help="Use this saved route for both seed and incumbent")
    parser.add_argument("--export-catalog-wide", action="store_true",
                        help=f"Save the validated result to {CATALOG_WIDE_RESULT}")
    args = parser.parse_args()
    if args.export_catalog_wide:
        if args.candidate_strategy != "catalog-wide":
            parser.error("--export-catalog-wide requires --candidate-strategy catalog-wide")
        if CATALOG_WIDE_RESULT.exists() or CATALOG_WIDE_RESULT.is_symlink():
            parser.error(f"{CATALOG_WIDE_RESULT} already exists")
    start_index = 12
    seed_options = {}
    if args.incumbent:
        saved_states = load_and_replay_baseline(args.incumbent)
        seed_options = {"seed_state": saved_states[start_index], "incumbent_state": saved_states[-1]}

    result = run_beam_search(
        start_index=start_index,
        beam_width=4,
        branch_factor=6,
        limit=2000,
        max_scans=64,
        partial_mining=True,
        candidate_strategy=args.candidate_strategy,
        **seed_options,
    )

    print("\n--- Rezultati final i Beam Search ---")

    print(
        "Asteroidet:",
        [stop.asteroid_id for stop in result.route]
    )

    print("Materialet:", result.collected_materials)
    print("Dita finale:", result.current_day)
    print("Delta-v:", result.remaining_dv)

    print("Score:", min(result.collected_materials))

    print("\n--- Validimi zyrtar ---")

    official_score = validate_route(
        result,
        verbose=True,
        require_full_route=bool(args.incumbent or args.export_catalog_wide),
    )

    print("\nScore zyrtar:", official_score)
    if args.export_catalog_wide:
        try:
            export_catalog_wide_result(result, official_score)
        except FileExistsError:
            parser.error(f"{CATALOG_WIDE_RESULT} already exists")
        print(f"Saved {CATALOG_WIDE_RESULT}")
