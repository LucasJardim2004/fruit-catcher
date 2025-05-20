import random

# GENETIC ALGORITHM

def create_individual(individual_size):
    """
    Creates a single individual (list of float genes) with values between -1 and 1.

    Args:
        individual_size (int): Number of genes in the individual.

    Returns:
        list[float]: The generated individual.
    """
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    """
    Generates a population of individuals.

    Args:
        individual_size (int): Number of genes per individual.
        population_size (int): Number of individuals in the population.

    Returns:
        list[list[float]]: The generated population.
    """
    return [create_individual(individual_size) for _ in range(population_size)]

def fitness_sort(population, fitness_function):
    """
    Evaluates and sorts the population by fitness in descending order.

    Args:
        population (list[list[float]]): The population to evaluate.
        fitness_function (function): The function used to evaluate fitness.

    Returns:
        list[tuple]: List of (individual, fitness) sorted by fitness.
    """
    scored = [(ind, fitness_function(ind)) for ind in population]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored

def select_elite(sorted_population, elite_rate):
    """
    Selects the top-performing individuals from the sorted population.

    Args:
        sorted_population (list[tuple]): Population sorted by fitness.
        elite_rate (float): Proportion of population to retain as elite.

    Returns:
        list[list[float]]: The elite individuals.
    """
    elite_count = max(1, int(len(sorted_population) * elite_rate))
    return [ind for ind, fit in sorted_population[:elite_count]]

def crossover(parent1, parent2):
    """
    Performs single-point crossover between two parents.

    Args:
        parent1 (list[float]): First parent.
        parent2 (list[float]): Second parent.

    Returns:
        list[float]: The resulting child.
    """
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate):
    """
    Applies mutation to an individual by adding small noise to each gene with some probability.

    Args:
        individual (list[float]): The individual to mutate.
        mutation_rate (float): Probability of mutating each gene.

    Returns:
        list[float]: The mutated individual.
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.uniform(-0.5, 0.5)
            individual[i] = max(-1, min(1, individual[i]))  # Clamp to [-1, 1]
    return individual

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations,
                      elite_rate=0.2, mutation_rate=0.05):
    """
    Runs a genetic algorithm to optimize a population of individuals.

    Args:
        individual_size (int): Number of genes per individual.
        population_size (int): Number of individuals in the population.
        fitness_function (function): Function to evaluate fitness of individuals.
        target_fitness (float): Target fitness value to stop the algorithm.
        generations (int): Maximum number of generations to run.
        elite_rate (float): Proportion of population to keep as elite (default: 0.2).
        mutation_rate (float): Probability of gene mutation (default: 0.05).

    Returns:
        tuple: The best individual found and its fitness score.
    """
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')

    for gen in range(generations):
        if gen % 10 == 0:  # A cada 10 gerações imprime progresso
            print(f"Generation {gen}/{generations} completed.")

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