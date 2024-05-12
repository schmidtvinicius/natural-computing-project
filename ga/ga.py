from abc import ABC, abstractmethod

import pandas as pd
import tqdm

class GeneticAlgorithm(ABC):
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        num_generations: int,
        dataset: pd.DataFrame,
        number_of_days: int = 45,
        seed: int = 42
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.seed = seed
        self.dataset = dataset

    @abstractmethod
    def initialize_population(self) -> list[list[tuple[str, int]]]:
        pass

    @abstractmethod
    def crossover(
        self,
        parent1: list[tuple[str, int]],
        parent2: list[tuple[str, int]]
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        pass

    @abstractmethod
    def mutate(self, individual: list[tuple[str, int]]) -> list[tuple[str, int]]:
        pass

    @abstractmethod
    def calculate_fitness(self, individual: list[tuple[str, int]]) -> float:
        pass

    def evolve(self, verbose: bool = False) -> list[tuple[str, int]]:
        population = self.initialize_population()

        for generation in range(self.num_generations) if not verbose else tqdm.tqdm(range(self.num_generations)):
            next_generation = []

            # Generate offspring until the new population size is reached
            while len(next_generation) < self.population_size:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population, False)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                next_generation.append(child1)
                next_generation.append(child2)

            population = next_generation

            # print best fitness in the current generation
            if verbose:
                print(f'Generation {generation + 1} best fitness score: {self.calculate_fitness(min(population, key=self.calculate_fitness))}')

        # Return the best individual from the final population
        return min(population, key=self.calculate_fitness)

    def select_parent(self, population: list[list[tuple[str, int]]], first: bool = True) -> list[tuple[str, int]]:
        # get best parent if first is True, else get second best parent
        return min(population, key=self.calculate_fitness) if first else sorted(population, key=self.calculate_fitness)[1]
