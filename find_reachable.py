
import runpy

esa = runpy.run_path("spoc-mining-mine-the-belt.py")

# E marrim objektin me t'dhenat e asteroideve
udp = esa["udp"]

# Asteroidi prej t'cilit nisem
start_id = 5

print("Asteroidi fillestar:", start_id)
print("Numri total i asteroideve:", len(udp.asteroids))

# provojna asteroide (disa)
import numpy as np
import pykep as pk

print("Numri i kandidateve qe po testohen:", 500)
targets = range(500)

departure_day = 0
arrival_day = 20

# Pozita edhe shpejtesia e asteroidit fillestar
r1, v1 = udp.asteroids[start_id].eph(
    esa["T_START"].mjd2000 + departure_day
)

reachable_asteroids = []
for target_id in targets:
    if target_id == start_id:
        continue

    # Pozita edhe shpejtesia e asteroidit ku dojna me shku
    r2, v2 = udp.asteroids[target_id].eph(
        esa["T_START"].mjd2000 + arrival_day
    )

    # Llogarisim trajektoren
    trajectory = pk.lambert_problem(
        r1=r1,
        r2=r2,
        tof=(arrival_day - departure_day) * pk.DAY2SEC,
        mu=udp.MU,
        cw=False,
        max_revs=0
    )

    dv1 = np.linalg.norm(np.array(v1) - np.array(trajectory.get_v1()[0]))
    dv2 = np.linalg.norm(np.array(v2) - np.array(trajectory.get_v2()[0]))

    total_dv = dv1 + dv2

    if total_dv <= 10000:
        material_id = int(udp.asteroid_materials_types[target_id])
        mass = float(udp.asteroid_masses[target_id])

        reachable_asteroids.append({
            "id": target_id,
            "dv": total_dv,
            "material": material_id,
            "mass": mass
        })
        print(f"Asteroidi {target_id}: {total_dv:.2f} m/s - I ARRITSHEM")
    else:
        print(f"Asteroidi {target_id}: {total_dv:.2f} m/s - I PAARRITSHEM")


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
    best_asteroid = min(
        reachable_asteroids,
        key=lambda asteroid: asteroid["dv"]
    )

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
    best_priority_asteroid = min(
        priority_asteroids,
        key=lambda asteroid: asteroid["dv"]
    )

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

r1, v1 = udp.asteroids[current_asteroid_id].eph(
    esa["T_START"].mjd2000 + current_day
)

second_reachable = []

for target_id in range(100):
    if target_id in [start_id, current_asteroid_id]:
        continue

    r2, v2 = udp.asteroids[target_id].eph(
        esa["T_START"].mjd2000 + next_arrival_day
    )

    trajectory = pk.lambert_problem(
        r1=r1,
        r2=r2,
        tof=(next_arrival_day - current_day) * pk.DAY2SEC,
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

    if total_dv <= remaining_dv:
        material_id = int(udp.asteroid_materials_types[target_id])
        mass = float(udp.asteroid_masses[target_id])

        second_reachable.append({
            "id": target_id,
            "dv": total_dv,
            "material": material_id,
            "mass": mass
        })

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
