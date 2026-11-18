import json
import time
from greedy_solver import run_greedy
from find_reachable import esa, udp

start_time = time.perf_counter()

result = run_greedy(
    start_id=300,
    fuel_threshold=5000,
    limit=2000
)

elapsed_time = time.perf_counter() - start_time

solution = (
    result["arrival_times"]
    + result["mining_times"]
    + result["visited_asteroids"]
)

chromosome = esa["convert_to_chromosome"](solution)
fitness = udp.fitness(chromosome, verbose=True)

baseline = {
    "algorithm": "Greedy + Grid Search",
    "start_id": 300,
    "fuel_threshold": 5000,
    "limit": 2000,
    "score": min(result["collected_materials"].values()),
    "official_score": float(fitness[0]),
    "visited_asteroids": result["visited_asteroids"],
    "arrival_times": result["arrival_times"],
    "mining_times": result["mining_times"],
    "collected_materials": result["collected_materials"],
    "remaining_dv": result["remaining_dv"],
    "final_day": result["current_day"],
    "runtime_seconds": elapsed_time
}

with open("baseline_result.json", "w") as file:
    json.dump(baseline, file, indent=4)

print("\nBaseline u ruajt me sukses!")
print("Score zyrtar:", fitness[0])