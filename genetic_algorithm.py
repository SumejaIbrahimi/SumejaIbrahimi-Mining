import random
from greedy_solver import run_greedy

POPULATION_SIZE = 10 #pe zgjedhim nje popullate te vogel njehere

def create_individual():
    return [
        random.randint(0, 9999),
        random.randint(2000, 7000),
        random.randint(1000, 10000)
    ]

def create_population():
    population = [[300, 5000, 2000]] # pasi qe eshte zgjidhja ma  e mire deri tani

    while len(population) < POPULATION_SIZE:
        population.append(create_individual())

    return population

def evaluate_fitness(individual):
    start_id, fuel_threshold, limit = individual #ndarja e gjeneve, ne python kjo sintakse quhet unpacking

    result = run_greedy( #ja dergojna parametrat greedyt qe ai me vazhdu misionin me keto
        start_id=start_id,
        limit=limit,
        fuel_threshold=fuel_threshold
    )

    score = min(result["collected_materials"].values())

    return score