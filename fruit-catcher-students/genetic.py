import random

def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

def genetic_algorithm(individual_size, population_size, fitness_function, max_score, generations, mutation_rate=0.01, elite_rate=0.1):
    population = [[random.uniform(-1, 1) for _ in range(individual_size)] for _ in range(population_size)]
    global_best = None  # Armazena o melhor indivíduo globalmente

    for generation in range(generations):
        # Avaliação da aptidão
        fitness_scores = [(individual, fitness_function(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        # Atualiza o melhor indivíduo global
        if global_best is None or fitness_scores[0][1] > global_best[1]:
            global_best = fitness_scores[0]

        # Exibe progresso a cada 10 gerações
        if generation % 10 == 0 or generation == generations - 1:
            best_score = fitness_scores[0][1]
            print(f"Geração {generation}: Melhor pontuação = {best_score}")

        # Verifica se atingiu a pontuação máxima
        if fitness_scores[0][1] >= max_score:
            break

        # Seleção dos melhores (elitismo)
        elite_size = int(elite_rate * population_size)
        new_population = [individual for individual, _ in fitness_scores[:elite_size]]

        # Cruzamento
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(new_population, k=2)
            crossover_point = random.randint(1, individual_size - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            new_population.append(child)

        # Mutação
        for individual in new_population[elite_size:]:
            if random.random() < mutation_rate:
                mutation_index = random.randint(0, individual_size - 1)
                individual[mutation_index] = random.uniform(-1, 1)

        population = new_population

    # Retorna o melhor indivíduo global após todas as gerações
    return global_best