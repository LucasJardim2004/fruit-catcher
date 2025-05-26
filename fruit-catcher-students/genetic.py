import random
import matplotlib.pyplot as plt

# Creates a random individual with weights between -1 and 1
def create_individual(individual_size):
    return [random.uniform(-1, 1) for _ in range(individual_size)]

# Generates a population of individuals
def generate_population(individual_size, population_size):
    return [create_individual(individual_size) for _ in range(population_size)]

# Main function implementing the Genetic Algorithm
def genetic_algorithm(individual_size,
                      population_size,
                      fitness_function,
                      max_score,
                      generations,
                      mutation_rate=0.05,
                      elite_rate=0.20,
                      seed=None):

    # Initialize the population randomly
    population = generate_population(individual_size, population_size)

    # Track the best individual globally
    global_best = (None, float('-inf'))

    # Lists to keep track of score statistics
    best_scores = []
    avg_scores = []

    # Main loop over generations
    for generation in range(generations):

        # Evaluate fitness of each individual
        scored = [(ind, fitness_function(ind, seed)) if seed is not None else (ind, fitness_function(ind)) for ind in population]

        # Sort individuals by fitness in descending order
        scored.sort(key=lambda x: x[1], reverse=True)

        # Extract the best individual of this generation
        best_ind, best_score = scored[0]

        # Update the global best if current one is better
        if best_score > global_best[1]:
            global_best = (best_ind.copy(), best_score)

        # Save best and average score for the generation
        best_scores.append(best_score)
        avg_scores.append(sum(score for _, score in scored) / len(scored))

        # Print status every 10 generations or at the last generation
        if generation % 10 == 0 or generation == generations - 1:
            print(f"Geração {generation}: Melhor = {best_score}")

        # Stop if the maximum score is reached
        if best_score >= max_score:
            break

        # Determine number of elite individuals to keep
        elite_size = max(1, int(elite_rate * population_size))

        # Copy elite individuals directly to the new population
        elites = [ind.copy() for ind, _ in scored[:elite_size]]

        # Prepare for crossover: extract individuals and their scores
        individuals, scores = zip(*scored)
        new_pop = elites.copy()

        # Create new individuals through crossover
        while len(new_pop) < population_size:
            p1, p2 = random.choices(individuals, weights=scores, k=2)  # Select two parents with probability based on fitness
            cp = random.randint(1, individual_size - 1)  # Choose a crossover point
            child = p1[:cp] + p2[cp:]  # Combine parts of parents
            new_pop.append(child)

        # Apply mutation to new individuals (excluding elites)
        for ind in new_pop[elite_size:]:
            if random.random() < mutation_rate:
                idx = random.randint(0, individual_size - 1)  # Select gene to mutate
                ind[idx] += random.gauss(0, 0.1)  # Add small Gaussian noise
                ind[idx] = max(min(ind[idx], 1), -1)  # Clamp values between -1 and 1

        # Set the new population for the next generation
        population = new_pop

    # Save training progress to a CSV file
    with open("training_progress.csv", "w") as f:
        f.write("generation,best_score,avg_score\n")
        for i in range(len(best_scores)):
            f.write(f"{i},{best_scores[i]},{avg_scores[i]}\n")

    # Plot the training progress
    plt.plot(best_scores, label="Melhor Score")      # Plot best score over generations
    plt.plot(avg_scores, label="Score Médio")        # Plot average score over generations
    plt.xlabel("Geração")                            # Label for x-axis
    plt.ylabel("Score")                              # Label for y-axis
    plt.title("Evolução do Treino")                  # Title of the plot
    plt.legend()                                     # Show legend
    plt.grid()                                       # Show grid
    plt.savefig("score_progress.png")                # Save the plot to a file
    # plt.show()                                     # Optionally show the plot on screen

    # Return the best individual and its score
    return global_best