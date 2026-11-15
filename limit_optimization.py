from greedy_solver import run_greedy
import time

limits = [500, 1000, 2000, 5000, 10000]

best_score = -1
best_limit = None

for limit in limits:
    print(f"\n--- Limit: {limit} ---")

    start_time = time.perf_counter()

    result = run_greedy(
        start_id=5,
        limit=limit,
        fuel_threshold=4000
    )

    elapsed_time = time.perf_counter() - start_time

    score = min(result["collected_materials"].values())

    print("Score:", score)
    print("Koha e ekzekutimit:", round(elapsed_time, 2), "sekonda")

    if score > best_score:
        best_score = score
        best_limit = limit

print("\n--- Rezultati me i mire ---")
print("Limiti me i mire:", best_limit)
print("Score maksimal:", best_score)