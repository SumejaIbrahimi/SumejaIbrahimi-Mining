
import numpy as np

# Me i lexu te dhenat e asteroideve
data = np.loadtxt("data/spoc/mining/candidates.txt")

# Numri i asteroideve(rreshta) edhe kolonave
print("Numri i asteroideve:", data.shape[0])
print("Numri i kolonave:", data.shape[1])

# materials esht dictionary dmth strukture e t'dhenave
materials = {
    0: "Gold",
    1: "Platinum",
    2: "Nickel",
    3: "Fuel"
}

# Me i numeru asteroidet sipas materialit
for code, name in materials.items():
    count = np.sum(data[:, 8] == code)
    print(f"{name}: {count} asteroide")

print("\n--- Analiza e mases ---")

for code, name in materials.items():

    # Zgjedhim vetem asteroidet e ketij materiali
    asteroids = data[data[:, 8] == code]

    # Marrim masat e tyre (kolona 7)
    masses = asteroids[:, 7]

    average_mass = np.mean(masses)
    maximum_mass = np.max(masses)

    # E gjejm asteroidin me masen me te madhe
    index = np.argmax(masses)
    asteroid_id = int(asteroids[index, 0])

    print(f"\nMateriali: {name}")
    print(f"Masa mesatare: {average_mass:.3f}")
    print(f"Masa maksimale: {maximum_mass:.3f}")
    print(f"Asteroidi me masen me te madhe: {asteroid_id}")
