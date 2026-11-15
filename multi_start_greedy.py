from greedy_solver import run_greedy

start_ids = [5, 100, 200, 300, 400]

best_score = -1
best_start = None

for start_id in start_ids:
    print(f"\n--- Start ID: {start_id} ---")

    result = run_greedy(start_id=start_id)

    score = min(result["collected_materials"].values())

    print("Score:", score)

    if score > best_score:
        best_score = score
        best_start = start_id

print("\n--- Rezultati me i mire ---")
print("Asteroidi fillestar:", best_start)
print("Score maksimal:", best_score)