import numpy as np
import pykep as pk
from dataclasses import dataclass
import math
import json

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

def validate_route(state, verbose=False):

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

if __name__ == "__main__":

    states = load_and_replay_baseline()

    start_state = states[12]
    final_state = states[-1]

    print("\n--- Gjendja fillestare per Beam Search ---")
    print("Asteroidi:", start_state.current_id)
    print("Dita:", start_state.current_day)
    print("Delta-v:", start_state.remaining_dv)
    print("Materialet:", start_state.collected_materials)

    print("\n--- Baseline i rikrijuar ---")
    print("Numri i gjendjeve:", len(states))
    print("Dita finale:", final_state.current_day)
    print("Delta-v:", final_state.remaining_dv)
    print("Materialet:", final_state.collected_materials)

    official_score = validate_route(
        final_state,
        verbose=True
    )

    print("\nScore zyrtar:", official_score)