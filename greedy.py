import math

def select_best_asteroid(candidates):
    if not candidates:
        return None

    return min(candidates, key=lambda asteroid: asteroid["dv"])

def select_priority_asteroid(candidates, mission, fuel_threshold=4000, improved=False):
    if not candidates:
        return None

    # Prioriteti i karburantit
    if mission["remaining_dv"] < fuel_threshold:
        fuel_candidates = [
            asteroid for asteroid in candidates
            if asteroid["material"] == 3
        ]

        if fuel_candidates:
            return select_best_asteroid(fuel_candidates)

    # Materiali qe kemi me se paku
    materials = mission["collected_materials"]
    min_amount = min(materials.values())

    needed_materials = [
        material for material, amount in materials.items()
        if amount == min_amount
    ]

    priority_candidates = [
        asteroid for asteroid in candidates
        if asteroid["material"] in needed_materials
    ]

    if priority_candidates:
        if improved:
            return max(
                priority_candidates,
                key=calculate_candidate_score
            )

        return select_best_asteroid(priority_candidates)

    return select_best_asteroid(candidates)

def calculate_candidate_score(asteroid):
    mass = asteroid["mass"]
    transfer_days = asteroid["transfer_days"]
    dv = asteroid["dv"]

    mining_days = max(1, math.ceil(mass * 30))
    total_days = transfer_days + mining_days

    fuel_penalty = (1 + dv / 10000) ** 4

    score = mass / (total_days * fuel_penalty)

    return score