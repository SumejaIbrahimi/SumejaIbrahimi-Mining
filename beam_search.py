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