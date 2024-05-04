from abc import ABC, abstractmethod

class GeneticAlgorithm(ABC):
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations, dataset):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.dataset = dataset

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, individual):
        pass

    @abstractmethod
    def calculate_fitness(self, individual):
        pass

    def evolve(self):
        population = self.initialize_population()

        for generation in range(self.num_generations):
            next_generation = []

            # Elitism: Keep the best individual from the current population
            next_generation.append(max(population, key=self.calculate_fitness))

            # Generate offspring until the new population size is reached
            while len(next_generation) < self.population_size:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population)
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                next_generation.append(offspring)

            population = next_generation

        # Return the best individual from the final population
        return max(population, key=self.calculate_fitness)

    @abstractmethod
    def select_parent(self, population):
        pass
