
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

targets = range(100)

departure_day = 0
arrival_day = 20

# Pozita edhe shpejtesia e asteroidit fillestar
r1, v1 = udp.asteroids[start_id].eph(
    esa["T_START"].mjd2000 + departure_day
)


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
