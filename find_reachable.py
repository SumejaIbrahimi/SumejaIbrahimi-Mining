
import runpy
import numpy as np
import pykep as pk
from greedy import select_best_asteroid

esa = runpy.run_path("spoc-mining-mine-the-belt.py")
# E marrim objektin me t'dhenat e asteroideve
udp = esa["udp"]

def find_reachable_asteroids(start_id, departure_day, arrival_day, available_dv, visited_asteroids, limit=500):
    reachable = []

    r1, v1 = udp.asteroids[start_id].eph(
        esa["T_START"].mjd2000 + departure_day
    )

    for target_id in range(min(limit, len(udp.asteroids))):
        if target_id == start_id or target_id in visited_asteroids:
            continue

        r2, v2 = udp.asteroids[target_id].eph(
            esa["T_START"].mjd2000 + arrival_day
        )

        trajectory = pk.lambert_problem(
            r1=r1,
            r2=r2,
            tof=(arrival_day - departure_day) * pk.DAY2SEC,
            mu=udp.MU,
            cw=False,
            max_revs=0
        )

        dv1 = np.linalg.norm(
            np.array(v1) - np.array(trajectory.get_v1()[0])
        )
        dv2 = np.linalg.norm(
            np.array(v2) - np.array(trajectory.get_v2()[0])
        )

        total_dv = dv1 + dv2

        if total_dv <= available_dv:
            reachable.append({
                "id": target_id,
                "dv": total_dv,
                "material": int(udp.asteroid_materials_types[target_id]),
                "mass": float(udp.asteroid_masses[target_id])
            })

    return reachable

def search_multiple_transfers(
    start_id,
    departure_day,
    available_dv,
    visited_asteroids,
    limit=500
):
    candidates = []

    transfer_days = [20, 40, 60, 80, 100]

    for days in transfer_days:
        arrival_day = departure_day + days

        if arrival_day > 1827:
            continue

        reachable = find_reachable_asteroids(
            start_id=start_id,
            departure_day=departure_day,
            arrival_day=arrival_day,
            available_dv=available_dv,
            visited_asteroids=visited_asteroids,
            limit=limit
        )

        for asteroid in reachable:
            asteroid["transfer_days"] = days
            asteroid["arrival_day"] = arrival_day

            candidates.append(asteroid)

    return candidates

# Asteroidi prej t'cilit nisem
start_id = 5

print("Asteroidi fillestar:", start_id)
print("Numri total i asteroideve:", len(udp.asteroids))

# provojna asteroide (disa)
print("Numri i kandidateve qe po testohen:", 500)

departure_day = 0
arrival_day = 20

reachable_asteroids = find_reachable_asteroids(
    start_id=start_id,
    departure_day=departure_day,
    arrival_day=arrival_day,
    available_dv=10000,
    visited_asteroids=[start_id],
    limit=500
)

# E analizojna asteroidin 52 se edhe pse ka kerku ma se paku delta-v,
# duhet me dit cfare materiali ka edhe masen
asteroid_id = 52

material_types = ["Gold", "Platinum", "Nickel", "Fuel"]

material_id = int(udp.asteroid_materials_types[asteroid_id])
mass = udp.asteroid_masses[asteroid_id]

print("\n--- Asteroidi 52 ---")
print("Materiali:", material_types[material_id])
print("Masa:", round(mass, 3))

# Simulojme nxjerrjen e nikelit per 10 dite
mining_days = 10

collected_mass = min(mass, mining_days / 30)

print("Ditet e nxjerrjes:", mining_days)
print("Sasia e nikelit e mbledhur:", round(collected_mass, 3))

print("\n--- Permbledhja e kerkimit ---")
print("Numri i asteroideve te arritshem:", len(reachable_asteroids))

for asteroid in reachable_asteroids:
    print(
        f"ID: {asteroid['id']}, "
        f"Delta-v: {asteroid['dv']:.2f} m/s, "
        f"Materiali: {asteroid['material']}, "
        f"Masa: {asteroid['mass']:.3f}"
    )

if reachable_asteroids:
    best_asteroid = select_best_asteroid(reachable_asteroids)

    print("\n--- Zgjedhja Greedy ---")
    print("Asteroidi i zgjedhur:", best_asteroid["id"])
    print("Delta-v:", round(best_asteroid["dv"], 2), "m/s")
    print("Masa:", round(best_asteroid["mass"], 3))

collected_materials = {
    0: 0.0,  # Gold
    1: 0.0,  # Platinum
    2: 0.0   # Nickel
}

print("\n--- Materialet e mbledhura ---")
print("Gold:", collected_materials[0])
print("Platinum:", collected_materials[1])
print("Nickel:", collected_materials[2])


min_amount = min(collected_materials.values())

needed_materials = [
    material_id
    for material_id, amount in collected_materials.items()
    if amount == min_amount
]

print("\n--- Materialet prioritare ---")
print("Materialet qe na duhen:", needed_materials)

priority_asteroids = [
    asteroid
    for asteroid in reachable_asteroids
    if asteroid["material"] in needed_materials
]

print("\n--- Asteroidet prioritare ---")

for asteroid in priority_asteroids:
    print(
        f"ID: {asteroid['id']}, "
        f"Materiali: {asteroid['material']}, "
        f"Delta-v: {asteroid['dv']:.2f} m/s"
    )

if priority_asteroids:
    best_priority_asteroid = select_best_asteroid(priority_asteroids)

    print("\n--- Zgjedhja Greedy me prioritet ---")
    print("Asteroidi:", best_priority_asteroid["id"])
    print("Materiali:", best_priority_asteroid["material"])
    print("Delta-v:", round(best_priority_asteroid["dv"], 2), "m/s")
    print("Masa:", round(best_priority_asteroid["mass"], 3))

if priority_asteroids:
    mining_days = 11

    material_id = best_priority_asteroid["material"]
    asteroid_mass = best_priority_asteroid["mass"]

    collected_mass = min(asteroid_mass, mining_days / 30)

    collected_materials[material_id] += collected_mass

    print("\n--- Nxjerrja e materialit ---")
    print("Asteroidi:", best_priority_asteroid["id"])
    print("Ditet e nxjerrjes:", mining_days)
    print("Sasia e nxjerre:", round(collected_mass, 3))

    print("\n--- Materialet pas nxjerrjes ---")
    print("Gold:", round(collected_materials[0], 3))
    print("Platinum:", round(collected_materials[1], 3))
    print("Nickel:", round(collected_materials[2], 3))

min_amount = min(collected_materials.values())

needed_materials = [
    material_id
    for material_id, amount in collected_materials.items()
    if amount == min_amount
]

print("\n--- Prioritetet pas nxjerrjes ---")
print("Materialet prioritare:", needed_materials)

initial_dv = 10000

remaining_dv = initial_dv - best_priority_asteroid["dv"]

print("\n--- Karburanti pas udhetimit ---")
print("Delta-v fillestare:", initial_dv, "m/s")
print("Delta-v e shpenzuar:", round(best_priority_asteroid["dv"], 2), "m/s")
print("Delta-v e mbetur:", round(remaining_dv, 2), "m/s")

current_asteroid_id = best_priority_asteroid["id"]

current_day = arrival_day + mining_days

next_arrival_day = current_day + 20

print("\n--- Pergatitja per udhetimin e dyte ---")
print("Asteroidi aktual:", current_asteroid_id)
print("Dita aktuale:", current_day)
print("Dita e mberritjes se ardhshme:", next_arrival_day)
print("Delta-v e disponueshme:", round(remaining_dv, 2), "m/s")

print("\n--- Kerkimi i asteroidit te dyte ---")

second_reachable = find_reachable_asteroids(
    start_id=current_asteroid_id,
    departure_day=current_day,
    arrival_day=next_arrival_day,
    available_dv=remaining_dv,
    visited_asteroids=[start_id, current_asteroid_id],
    limit=100
)

print("Numri i asteroideve te arritshem:", len(second_reachable))

print("\n--- Kandidatet per asteroidin e dyte ---")

material_names = ["Gold", "Platinum", "Nickel", "Fuel"]

for asteroid in second_reachable:
    print(
        f"ID: {asteroid['id']}, "
        f"Materiali: {material_names[asteroid['material']]}, "
        f"Masa: {asteroid['mass']:.3f}, "
        f"Delta-v: {asteroid['dv']:.2f} m/s"
    )

second_priority_asteroids = [
    asteroid
    for asteroid in second_reachable
    if asteroid["material"] in needed_materials
]

print("\n--- Kontrolli i prioriteteve ---")

if second_priority_asteroids:
    print("U gjeten asteroide me materiale prioritare.")
else:
    print("Nuk u gjet asnje asteroid me Gold ose Platinum.")
    print("Duhet te zgjerojme kerkimin.")

print("\n--- Testimi i funksionit automatik ---")

test_reachable = find_reachable_asteroids(
    start_id=52,
    departure_day=31,
    arrival_day=51,
    available_dv=remaining_dv,
    visited_asteroids=[5, 52],
    limit=500
)

print("Numri i asteroideve te arritshem:", len(test_reachable))

for asteroid in test_reachable:
    print(
        f"ID: {asteroid['id']}, "
        f"Materiali: {asteroid['material']}, "
        f"Delta-v: {asteroid['dv']:.2f} m/s"
    )

next_priority_asteroids = [
    asteroid
    for asteroid in test_reachable
    if asteroid["material"] in needed_materials
]

print("\n--- Zgjedhja e asteroidit te dyte ---")

if next_priority_asteroids:
    next_asteroid = select_best_asteroid(next_priority_asteroids)

    print("Asteroidi i zgjedhur:", next_asteroid["id"])
    print("Materiali:", next_asteroid["material"])
    print("Delta-v:", round(next_asteroid["dv"], 2), "m/s")
    print("Masa:", round(next_asteroid["mass"], 3))
else:
    print("Nuk ka asteroide me materiale prioritare.")

if next_priority_asteroids:
    remaining_dv -= next_asteroid["dv"]

    print("\n--- Gjendja pas udhetimit te dyte ---")
    print("Asteroidi aktual:", next_asteroid["id"])
    print("Delta-v e mbetur:", round(remaining_dv, 2), "m/s")

visited_asteroids = [
    start_id,
    best_priority_asteroid["id"]
]

if next_priority_asteroids:
    visited_asteroids.append(next_asteroid["id"])

print("\n--- Asteroidet e vizituar ---")
print(visited_asteroids)

# Gjendja aktuale e ketij misioni
mission_current_id = next_asteroid["id"]
mission_current_day = next_arrival_day
mission_remaining_dv = remaining_dv

print("\n--- Gjendja aktuale e misionit ---")
print("Asteroidi aktual:", mission_current_id)
print("Dita aktuale:", mission_current_day)
print("Delta-v e mbetur:", round(mission_remaining_dv, 2))
print("Asteroidet e vizituar:", visited_asteroids)

# Mining i materialit ne asteroidin aktual
mining_days = 23

current_material = next_asteroid["material"]
current_mass = next_asteroid["mass"]

mined_mass = min(current_mass, mining_days / 30.0)

if current_material in collected_materials:
    collected_materials[current_material] += mined_mass

mission_current_day += mining_days

print("\n--- Rezultati i minimit ---")
print("Asteroidi:", mission_current_id)
print("Materiali:", current_material)
print("Masa e mbledhur:", round(mined_mass, 3))
print("Dita pas minimit:", mission_current_day)
print("Materialet e mbledhura:", collected_materials)

# Perditesimi i materialeve qe na mungojne (platinum n'kete rast)
needed_materials = [
    material
    for material in [0, 1, 2]
    if collected_materials[material] == 0
]

print("\n--- Materialet qe na mungojne ---")
print("Materialet:", needed_materials)

material_names = {
    0: "Gold",
    1: "Platinum",
    2: "Nickel"
}

for material in needed_materials:
    print("Duhet te kerkojme:", material_names[material])

next_departure_day = mission_current_day
next_arrival_day = next_departure_day + 20

future_reachable = find_reachable_asteroids(
    start_id=mission_current_id,
    departure_day=next_departure_day,
    arrival_day=next_arrival_day,
    available_dv=mission_remaining_dv,
    visited_asteroids=visited_asteroids,
    limit=500
)

print("\n--- Kerkimi i ardhshem ---")
print("Dita e nisjes:", next_departure_day)
print("Dita e mberritjes:", next_arrival_day)
print("Delta-v ne dispozicion:", round(mission_remaining_dv, 2))
print("Numri i asteroideve te arritshem:", len(future_reachable))

for asteroid in future_reachable:
    print(asteroid)

# Numri i asteroidve te arritshem na ka dal 0, per shkak te kushteve qe i kena vendos

# Validimi i trajektores aktuale me evaluatorin e ESA-s

solution = [
    0, 20, 51,    # Kohet e mberritjes
    0, 11, 23,    # Kohet e minimit
    5, 52, 162    # ID-te e asteroideve
]

chromosome = esa["convert_to_chromosome"](solution)

print("\n--- Validimi me evaluatorin e ESA-s ---")

result = udp.fitness(chromosome, verbose=True)

print("Score zyrtar:", result[0])
print("\n--- Kerkimi me kohe te ndryshme ---")

multiple_candidates = search_multiple_transfers(
    start_id=5,
    departure_day=1,
    available_dv=10000,
    visited_asteroids=[5],
    limit=500
)

print("Numri i alternativave:", len(multiple_candidates))

best_transfer = select_best_asteroid(multiple_candidates)

print("Alternativa me Delta-v minimale:")
print(best_transfer)