import json
import time
from beam_search import run_beam_search, validate_route

start_time = time.perf_counter()

result = run_beam_search(
    start_index=12,
    beam_width=8,
    branch_factor=6,
    limit=2000,
    max_scans=128
)

official_score = validate_route(result)

elapsed = time.perf_counter() - start_time

data = {
    "algorithm": "Beam Search",
    "start_index": 12,
    "beam_width": 8,
    "branch_factor": 6,
    "limit": 2000,
    "max_scans": 128,
    "partial_mining": True,
    "score": min(result.collected_materials),
    "official_score": official_score,
    "visited_asteroids": [
        stop.asteroid_id for stop in result.route
    ],
    "arrival_times": [
        stop.arrival_day for stop in result.route
    ],
    "mining_times": [
        stop.mining_days for stop in result.route
    ],
    "collected_materials": {
        str(i): amount
        for i, amount in enumerate(result.collected_materials)
    },
    "remaining_dv": result.remaining_dv,
    "final_day": result.current_day,
    "runtime_seconds": elapsed
}

with open("beam_best_result.json", "w") as file:
    json.dump(data, file, indent=4)

print("Rezultati u ruajt me sukses!")
print("Score zyrtar:", official_score)