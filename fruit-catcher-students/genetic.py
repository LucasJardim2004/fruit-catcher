import random


def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]


def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]


def genetic_algorithm(individual_size,
                      population_size,
                      fitness_function,
                      max_score,
                      generations,
                      mutation_rate=0.01,
                      elite_rate=0.1):
    # Inicializa população
    population = generate_population(individual_size, population_size)
    # Guarda o melhor indivíduo global como (indivíduo, pontuação)
    global_best = (None, float('-inf'))

    for generation in range(generations):
        # Avaliação da aptidão (fitness)
        fitness_scores = [(ind, fitness_function(ind)) for ind in population]
        # Ordena pelo melhor fitness (descendente)
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_individual, best_score = fitness_scores[0]
        # Atualiza o melhor global
        if best_score > global_best[1]:
            # Copia para não alterar acidentalmente o individuo original
            global_best = (best_individual.copy(), best_score)

        # Log a cada 10 gerações e na última
        if generation % 10 == 0 or generation == generations - 1:
            print(f"Geração {generation}: Melhor pontuação = {best_score}")

        # Critério de paragem se atinge max_score
        if best_score >= max_score:
            break

        # Elitismo: copia os top N
        elite_size = max(1, int(elite_rate * population_size))
        elites = [ind.copy() for ind, _ in fitness_scores[:elite_size]]

        # Seleção cruzamento e mutação
        new_population = elites.copy()
        # Prepara listas para seleção com pesos
        individuals, scores = zip(*fitness_scores)

        # Geração de filhos até encher a população
        while len(new_population) < population_size:
            # Seleção com probabilidade proporcional ao fitness
            parent1, parent2 = random.choices(individuals, weights=scores, k=2)
            # Crossover de um ponto
            cp = random.randint(1, individual_size - 1)
            child = parent1[:cp] + parent2[cp:]
            new_population.append(child)

        # Mutação nos filhos (excluindo elites)
        for ind in new_population[elite_size:]:
            if random.random() < mutation_rate:
                idx = random.randint(0, individual_size - 1)
                ind[idx] = random.uniform(-1, 1)

        population = new_population

    # Retorna o melhor encontrado (indivíduo, pontuação)
    return global_best