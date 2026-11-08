
import runpy

# Me mar kodin zyrtar te ESA-s
esa = runpy.run_path("spoc-mining-mine-the-belt.py")

# Marrim funksionin dhe objektin e vleresimit
convert = esa["convert_to_chromosome"]
udp = esa["udp"]

# Nje zgjidhje e thjeshte per test veq
solution = [
    0, 20, 55,     # Kohet e mberritjes
    0, 10, 15,     # Ditet e nxjerrjes
    5, 42, 91      # ID-te e asteroideve
]

# E kthejna ne formatin e kerkuar nga ESA
chromosome = convert(solution)

print("Gjatesia e chromosome:", len(chromosome))

print("Kapaciteti fillestar i anijes: 10000 m/s delta-v")
# Vlersohet zgjidhja
result = udp.fitness(chromosome, verbose=True)

print("Rezultati:", result[0])

import numpy as np
import pykep as pk

# Asteroidi fillestar dhe ai ku dojna me shku
start_id = 5
target_id = 42

# Koha e nisjes dhe e mberritjes
departure_day = 0
arrival_day = 20

# Pozitat dhe shpejtesite e asteroideve
r1, v1 = udp.asteroids[start_id].eph(
    esa["T_START"].mjd2000 + departure_day
)

r2, v2 = udp.asteroids[target_id].eph(
    esa["T_START"].mjd2000 + arrival_day
)

# Llogarisim trajektoren e udhetimit
trajectory = pk.lambert_problem(
    r1=r1,
    r2=r2,
    tof=(arrival_day - departure_day) * pk.DAY2SEC,
    mu=udp.MU,
    cw=False,
    max_revs=0
)

# Llogarisim delta-v
dv1 = np.linalg.norm(np.array(v1) - np.array(trajectory.get_v1()[0]))
dv2 = np.linalg.norm(np.array(v2) - np.array(trajectory.get_v2()[0]))

total_dv = dv1 + dv2

print("\nDelta-v e nevojshme:", round(total_dv, 2), "m/s")
print("Delta-v ne dispozicion: 10000 m/s")

