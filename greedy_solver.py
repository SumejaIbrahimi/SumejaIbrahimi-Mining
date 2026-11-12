from mission import initialize_mission, mine_asteroid
from find_reachable import search_multiple_transfers, esa, udp
from greedy import select_priority_asteroid
import math


def run_greedy(start_id=5, limit=500):
    mission = initialize_mission(start_id)

    while mission["current_day"] < 1827:

        candidates = search_multiple_transfers(
            start_id=mission["current_id"],
            departure_day=mission["current_day"],
            available_dv=mission["remaining_dv"],
            visited_asteroids=mission["visited_asteroids"],
            limit=limit
        )

        if not candidates:
            print("Nuk ka me asteroide te arritshem.")
            break

        # Kontrollojme qe edhe minimi te perfundoje me kohe
        candidates = [
            asteroid for asteroid in candidates
            if asteroid["arrival_day"] + max(
                1, math.ceil(asteroid["mass"] * 30)
            ) <= 1827
        ]

        if not candidates:
            break

        best = select_priority_asteroid(candidates, mission)

        mining_days = max(1, math.ceil(best["mass"] * 30))

        # Perditesimi pas udhetimit
        mission["remaining_dv"] -= best["dv"]
        mission["current_id"] = best["id"]
        mission["current_day"] = best["arrival_day"]

        mission["visited_asteroids"].append(best["id"])
        mission["arrival_times"].append(best["arrival_day"])
        mission["mining_times"].append(mining_days)

        # Minimi
        mine_asteroid(mission, best, mining_days)

        print(
            f"Asteroidi: {best['id']} | "
            f"Materiali: {best['material']} | "
            f"Dita: {mission['current_day']} | "
            f"Delta-v: {mission['remaining_dv']:.2f}"
        )

    return mission


if __name__ == "__main__":
    result = run_greedy()

    print("\n--- Rezultati i misionit ---")
    print("Asteroidet:", result["visited_asteroids"])
    print("Materialet:", result["collected_materials"])
    print("Score:", min(result["collected_materials"].values()))

       # Krijimi i zgjidhjes n'formatin e ESA-s
    solution = (
        result["arrival_times"]
        + result["mining_times"]
        + result["visited_asteroids"]
    )

    chromosome = esa["convert_to_chromosome"](solution)

    print("\n--- Validimi zyrtar i Greedy ---")

    fitness = udp.fitness(chromosome, verbose=True)

    print("Score zyrtar:", fitness[0]) 