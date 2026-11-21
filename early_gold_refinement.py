"""Refine the gold stop before the full refuel at asteroid 1850."""

import math
from pathlib import Path
import sys

from early_gold_experiment import run_worker, search


SOURCE = Path("middle_gold_best_result.json")
OUTPUT = Path("early_gold_refinement_best_result.json")
SCHEDULES = ((207, 30), (208, 29))
MIN_YIELD = 0.905321
CANDIDATE_LIMIT = 12
SOLVE_LIMIT = 114


def gold_candidates(udp, visited, epoch, mining_days):
    epochs = (epoch, epoch + mining_days)
    reference = [udp.asteroids[4745].eph(day) for day in epochs]
    masses = udp.asteroid_masses
    mining_capacity = mining_days / 30

    def geometry_rank(asteroid_id):
        cost = 0.0
        for day, (position, velocity) in zip(epochs, reference):
            target_position, target_velocity = udp.asteroids[asteroid_id].eph(day)
            cost += math.hypot(math.dist(position, target_position) / (57 * 86400),
                               math.dist(velocity, target_velocity))
        return cost, -min(masses[asteroid_id], mining_capacity), asteroid_id

    ranked = sorted(
        (i for i, material in enumerate(udp.asteroid_materials_types)
         if i not in visited and material == 0 and min(masses[i], mining_capacity) >= MIN_YIELD),
        key=geometry_rank,
    )
    nearby = ranked[:CANDIDATE_LIMIT - 3]
    richer = sorted(ranked[len(nearby):], key=lambda i: (-min(masses[i], mining_capacity), i))
    return nearby + richer[:3]


def run():
    search(
        SOURCE, OUTPUT, schedules=SCHEDULES, expected_score=4.320457054847707,
        inner_stops=((4745, 210, 27), (772, 294, 8)), candidate_limit=CANDIDATE_LIMIT,
        solve_limit=SOLVE_LIMIT, select_candidates=gold_candidates,
        algorithm="Early gold refinement",
    )


if __name__ == "__main__":
    sys.exit(run_worker("early_gold_refinement"))
