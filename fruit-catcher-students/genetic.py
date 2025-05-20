import random

#GENETIC ALGORITHM

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def fitness_sort(population, fitness_function):
    scored = [(ind, fitness_function(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def select_elite(sorted_population, elite_rate):
    elite_count = max(1, int(len(sorted_population) * elite_rate))
    return [ind for ind, fit in sorted_population[:elite_count]]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.uniform(-0.5, 0.5)
            individual[i] = max(-1, min(1, individual[i]))  # Clamp to [-1,1]
    return individual

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2,
                      mutation_rate=0.05):
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')

    for gen in range(generations):
        scored_population = fitness_sort(population, fitness_function)

        # Update best individual
        if scored_population[0][1] > best_fitness:
            best_individual, best_fitness = scored_population[0]

        if best_fitness >= target_fitness:
            print(f"Target fitness reached at generation {gen}")
            break

        elite = select_elite(scored_population, elite_rate)

        # Generate new population
        new_population = elite.copy()

        while len(new_population) < population_size:
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_individual, best_fitness