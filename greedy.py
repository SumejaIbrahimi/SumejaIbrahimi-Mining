
def select_best_asteroid(candidates):
    if not candidates:
        return None

    return min(candidates, key=lambda asteroid: asteroid["dv"])

def select_priority_asteroid(candidates, mission, fuel_threshold=4000):
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
        return select_best_asteroid(priority_candidates)

    return select_best_asteroid(candidates)