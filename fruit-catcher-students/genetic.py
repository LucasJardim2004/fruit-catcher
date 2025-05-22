import random

def create_individual(individual_size):
    """
    Cria um indivíduo com pesos aleatórios entre -1 e 1.

    Args:
        individual_size (int): Número de pesos no indivíduo.

    Returns:
        list[float]: Lista de pesos aleatórios.
    """
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    """
    Gera uma população inicial de indivíduos.

    Args:
        individual_size (int): Tamanho de cada indivíduo.
        population_size (int): Número de indivíduos na população.

    Returns:
        list[list[float]]: Lista com a população de indivíduos.
    """
    return [create_individual(individual_size) for _ in range(population_size)]

def genetic_algorithm(individual_size,
                      population_size,
                      fitness_function,
                      max_score,
                      generations,
                      mutation_rate=0.05,
                      elite_rate=0.20,
                      seed=None):
    """
    Executa um algoritmo genético para otimizar os pesos de uma rede neural.

    Args:
        individual_size (int): Número de pesos em cada indivíduo.
        population_size (int): Número de indivíduos na população.
        fitness_function (callable): Função que avalia a performance de um indivíduo.
                                     Pode aceitar um argumento opcional `seed`.
        max_score (float): Pontuação alvo para encerrar o treino.
        generations (int): Número máximo de gerações.
        mutation_rate (float, optional): Probabilidade de mutação. Default é 0.05.
        elite_rate (float, optional): Percentagem da população que é mantida sem alterações. Default é 0.20.
        seed (Any, optional): Valor adicional passado para a `fitness_function`, se necessário.

    Returns:
        tuple[list[float], float]: O melhor indivíduo encontrado e a sua pontuação.
    """
    # Inicializa população
    population = generate_population(individual_size, population_size)
    global_best = (None, float('-inf'))

    for generation in range(generations):
        # Avalia fitness
        scored = [(ind, fitness_function(ind, seed)) if seed is not None else (ind, fitness_function(ind)) for ind in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_ind, best_score = scored[0]

        # Atualiza global
        if best_score > global_best[1]:
            global_best = (best_ind.copy(), best_score)

        if generation % 10 == 0 or generation == generations - 1:
            print(f"Geração {generation}: Melhor = {best_score}")

        if best_score >= max_score:
            break

        # Elitismo
        elite_size = max(1, int(elite_rate * population_size))
        elites = [ind.copy() for ind, _ in scored[:elite_size]]

        # Seleção proporcional e crossover
        individuals, scores = zip(*scored)
        new_pop = elites.copy()
        while len(new_pop) < population_size:
            p1, p2 = random.choices(individuals, weights=scores, k=2)
            cp = random.randint(1, individual_size - 1)
            child = p1[:cp] + p2[cp:]
            new_pop.append(child)

        # Mutação (só nos não-elites)
        for ind in new_pop[elite_size:]:
            if random.random() < mutation_rate:
                idx = random.randint(0, individual_size - 1)
                ind[idx] += random.gauss(0, 0.1)
                ind[idx] = max(min(ind[idx], 1), -1)  # clamp opcional

        population = new_pop

    return global_best  # (melhor_indivíduo, pontuação)