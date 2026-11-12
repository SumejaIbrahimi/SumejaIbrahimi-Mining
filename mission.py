def initialize_mission(start_id=5):
    return {
        "current_id": start_id,
        "current_day": 1.0,
        "remaining_dv": 10000.0,

        "collected_materials": {
            0: 0.0,
            1: 0.0,
            2: 0.0
        },

        "visited_asteroids": [start_id],
        "arrival_times": [0.0],
        "mining_times": [1.0]
    }

def mine_asteroid(mission, asteroid, mining_days):
    material = asteroid["material"]
    mass = asteroid["mass"]

    mined_mass = min(mass, mining_days / 30.0) # sasia qe arrijm me nxjerr

    if material == 3: # fuel
        mission["remaining_dv"] = min(
            10000.0,
            mission["remaining_dv"] + mined_mass * 10000
        )
    else:
        mission["collected_materials"][material] += mined_mass

    mission["current_day"] += mining_days

    return mined_mass