from abc import ABC, abstractmethod
from typing import List, Dict, Any

import pandas as pd

class GeneticAlgorithm(ABC):
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        num_generations: int,
        dataset: pd.DataFrame
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.dataset = dataset

    @abstractmethod
    def initialize_population(self) -> list[list[tuple[str, int]]]:
        pass

    @abstractmethod
    def crossover(
        self,
        parent1: list[tuple[str, int]],
        parent2: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        pass

    @abstractmethod
    def mutate(self, individual: list[tuple[str, int]]) -> list[tuple[str, int]]:
        pass

    @abstractmethod
    def calculate_fitness(self, individual: list[tuple[str, int]]) -> float:
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
    def select_parent(self, population: list[list[tuple[str, int]]]) -> list[tuple[str, int]]:
        pass
