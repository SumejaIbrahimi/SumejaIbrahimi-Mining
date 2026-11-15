from greedy_solver import run_greedy

fuel_thresholds = [2000, 3000, 4000, 5000, 6000, 7000]

best_score = -1
best_threshold = None

for threshold in fuel_thresholds:
    print(f"\n--- Fuel threshold: {threshold} ---")

    result = run_greedy(
        start_id=5,
        fuel_threshold=threshold
    )

    score = min(result["collected_materials"].values())

    print("Score:", score)

    if score > best_score:
        best_score = score
        best_threshold = threshold

print("\n--- Rezultati me i mire ---")
print("Pragu optimal i testuar:", best_threshold)
print("Score maksimal:", best_score)