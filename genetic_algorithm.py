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

#selection
def select_parent(population, fitness_scores):
    tournament = random.sample(range(len(population)), 3)

    best_index = max(
        tournament,
        key=lambda i: fitness_scores[i]
    )

    return population[best_index].copy() 
# me copy sepse me kthy nje kopje te individit fitues,
# qe mos me ndryshu aksidentalisht individin origjinal gjate mutacioni

#crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, 2)

    child = (
        parent1[:crossover_point] # slicing, marrja e nje pjese te listes
        + parent2[crossover_point:]
    )

    return child

# mutation
def mutate(individual, mutation_rate=0.2): # secili gjen ka 20% mundesi te ndryshoj PSH
    if random.random() < mutation_rate:
        individual[0] = random.randint(0, 9999)

    if random.random() < mutation_rate:
        individual[1] = random.randint(2000, 7000)

    if random.random() < mutation_rate:
        individual[2] = random.randint(1000, 10000)

    return individual

def run_genetic_algorithm(generations=5):
    random.seed(42)

    population = create_population()

    best_individual = None
    best_score = -1

    for generation in range(generations):

        fitness_scores = [
            evaluate_fitness(individual)
            for individual in population
        ]

        generation_best_index = max(
            range(len(population)),
            key=lambda i: fitness_scores[i]
        )

        generation_best_score = fitness_scores[
            generation_best_index
        ]

        if generation_best_score > best_score:
            best_score = generation_best_score
            best_individual = population[
                generation_best_index
            ].copy()

        print(f"\n--- Gjenerata {generation + 1} ---")
        print("Score i gjenerates:", generation_best_score)
        print("Score me i mire:", best_score)
        print("Individi me i mire:", best_individual)

        new_population = [best_individual.copy()]

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(population, fitness_scores)
            parent2 = select_parent(population, fitness_scores)

            child = crossover(parent1, parent2)
            child = mutate(child)

            new_population.append(child)

        population = new_population

    return best_individual, best_score

if __name__ == "__main__":
    best_individual, best_score = run_genetic_algorithm()

    print("\n--- Rezultati final i GA ---")
    print("Parametrat me te mire:", best_individual)
    print("Score maksimal:", best_score)