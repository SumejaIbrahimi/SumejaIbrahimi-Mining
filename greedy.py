
def select_best_asteroid(candidates):
    if not candidates:
        return None

    return min(candidates, key=lambda asteroid: asteroid["dv"])
