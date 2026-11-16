from greedy_solver import run_greedy
import time

# po dojna me i testu parametrat te kombinuar
# metoda e perdorur ketu eshte grid search, na e gjen kombinimin me te mire nder parametrat e testuar
start_ids = [5, 100, 200, 300, 400]
fuel_thresholds = [3000, 4000, 5000]
limits = [2000, 5000, 10000]

best_score = -1
best_parameters = None

for start_id in start_ids:
    for threshold in fuel_thresholds:
        for limit in limits:

            print(f"\nStart: {start_id} | Fuel: {threshold} | Limit: {limit}")

            start_time = time.perf_counter()

            result = run_greedy(
                start_id=start_id,
                limit=limit,
                fuel_threshold=threshold
            )

            elapsed_time = time.perf_counter() - start_time

            score = min(result["collected_materials"].values())

            print("Score:", score)
            print("Koha:", round(elapsed_time, 2), "sekonda")

            if score > best_score:
                best_score = score
                best_parameters = (start_id, threshold, limit)

print("\n--- Rezultati me i mire ---")
print("Parametrat me te mire:", best_parameters)
print("Score maksimal:", best_score)