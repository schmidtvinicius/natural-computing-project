from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import tqdm

class GeneticAlgorithm(ABC):
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        num_generations: int,
        dataset: pd.DataFrame,
        elsitism: bool = True,
        seed: int = 42
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.elsitism = elsitism
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

    def evolve(
        self,
        verbose: bool = False,
        plot_generations: bool = False,
        return_generations_scores: bool = False
    ) -> list[tuple[str, int]] | tuple[list[tuple[str, int]], list[float]]:
        population = self.initialize_population()

        if plot_generations or return_generations_scores:
            best_scores = []

        for generation in range(self.num_generations) if not verbose else tqdm.tqdm(range(self.num_generations)):
            next_generation = []

            # Generate offspring until the new population size is reached
            while len(next_generation) < self.population_size:
                parent1 = self.select_parent(population)
                parent2 = self.select_parent(population, False)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                # compare parent and child fitness scores
                if self.elsitism:
                    if self.calculate_fitness(parent1) < self.calculate_fitness(child1):
                        next_generation.append(parent1)
                    else:
                        next_generation.append(child1)
                    if self.calculate_fitness(parent2) < self.calculate_fitness(child2):
                        next_generation.append(parent2)
                    else:
                        next_generation.append(child2)
                else:
                    next_generation.append(child1)
                    next_generation.append(child2)

            population = next_generation

            # print best fitness in the current generation
            if verbose:
                print(f'Generation {generation + 1} best fitness score: {self.calculate_fitness(min(population, key=self.calculate_fitness))}')

            if plot_generations or return_generations_scores:
                best_scores.append(self.calculate_fitness(min(population, key=self.calculate_fitness)))

        if plot_generations:
            self.plot_evolution(best_scores)

        if return_generations_scores:
            return min(population, key=self.calculate_fitness), best_scores
        return min(population, key=self.calculate_fitness)
    
    def multiple_runs(self, num_runs: int) -> None:
        """"
        Run the genetic algorithm multiple times and plot the average fitness score over generations
        """
        avg_best_scores = []
        for _ in range(num_runs):
            _, best_scores = self.evolve(verbose=True,return_generations_scores=True)
            avg_best_scores.append(best_scores)
        
        avg_best_scores = [sum(x) / num_runs for x in zip(*avg_best_scores)]
        self.plot_evolution(avg_best_scores)

    def select_parent(self, population: list[list[tuple[str, int]]], first: bool = True) -> list[tuple[str, int]]:
        # get best parent if first is True, else get second best parent
        return min(population, key=self.calculate_fitness) if first else sorted(population, key=self.calculate_fitness)[1]
    
    def plot_evolution(self, best_scores: list[float]) -> None:

        plt.plot(best_scores)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('Fitness Score over Generations')
        plt.savefig('fitness_score.png')
