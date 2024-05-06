import pandas as pd
import numpy as np
import random

from ga import GeneticAlgorithm

class GeneticAlgorithmString(GeneticAlgorithm):
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        num_generations: int,
        dataset: pd.DataFrame,
        seed: int = 42
    ):
        super().__init__(
            population_size,
            mutation_rate,
            crossover_rate,
            num_generations,
            dataset, 
            seed
        )

    def initialize_individual(self, airports: np.ndarray, total_days: int) -> str:
        np.random.shuffle(airports)
        individual = ''
        for index, airport in enumerate(airports):
            days = np.random.randint(low=2,high=4)
            while(len(individual)/3 + days) > total_days:
                if index == len(airports) - 1:
                    days = total_days - len(individual)
                else:
                    days = np.random.randint(low=2,high=6)
            individual += airport * days
        
        return individual


    def initialize_population(self) -> list[str]:
        pass


    def crossover(self, parent1: str, parent2: str) -> str:
        # Placeholder for crossover method
        pass


    def mutate(self, individual: str) -> str:
        # Placeholder for mutation method
        pass


    def calculate_fitness(self, individual: str) -> float:
        # Placeholder for fitness calculation method
        pass


    def select_parent(self, population: list[str]) -> str:
        # Placeholder for parent selection method
        pass
