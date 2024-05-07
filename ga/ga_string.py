import pandas as pd
import numpy as np
import random
import re

from collections import Counter
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


    def initialize_individual(self, airports: np.ndarray, days_per_city: int) -> str:
        np.random.shuffle(airports)
        return ''.join([airport * days_per_city for airport in airports])


    def initialize_population(self) -> list[str]:
        population = []
        total_days = len(self.dataset['flightDate'].unique())
        airports = self.dataset['startingAirport'].unique()
        days_per_city = int(total_days/len(airports))

        for _ in np.arange(self.population_size):
            population.append(self.initialize_individual(np.copy(airports), days_per_city))
        
        return population


    def crossover(self, parent1: str, parent2: str) -> list[str]:
        if len(parent1) != len(parent2):
            raise RuntimeError('Parents should have the same length')
    

        days_parent1 = Counter(re.findall('[A-Z]{3}', parent1))
        days_parent2 = Counter(re.findall('[A-Z]{3}', parent2))
        order_parent1 = list(days_parent1.keys())
        order_parent2 = list(days_parent2.keys())

        cut_point1 = np.random.randint(low=0,high=len(order_parent1) - 1)
        cut_point2 = np.random.randint(low=0,high=len(order_parent1) - 1)

        if cut_point1 > cut_point2:
            cut_point1, cut_point2 = cut_point2, cut_point1

        child1 = order_parent1[cut_point1:cut_point2]
        child2 = order_parent2[cut_point1:cut_point2]

        missing_cities1 = list(set(order_parent2) - set(child1))
        missing_cities2 = list(set(order_parent1) - set(child2))

        child1 = missing_cities1[len(order_parent1) - cut_point2:] + child1 + missing_cities1[:len(order_parent1) - cut_point2]
        child2 = missing_cities2[len(order_parent2) - cut_point2:] + child2 + missing_cities2[:len(order_parent2) - cut_point2]

        child1 = ''.join([airport * days_parent1.get(airport) for airport in child1])
        child2 = ''.join([airport * days_parent2.get(airport) for airport in child2])

        return [child1,child2]


    def mutate(self, individual: str) -> str:
        days_per_airport = Counter(Counter(re.findall('[A-Z]{3}', individual)))
        airport_order = list(days_per_airport.keys())
        previous_change = 0

        for index, airport in enumerate(airport_order):
            if previous_change != 0:
                days_per_airport[airport] += previous_change
                previous_change = 0
                continue
            if np.random.rand() < self.mutation_rate:
                print(f'mutating {airport}')
                new_days = np.random.randint(low=2,high=5)
                while new_days == days_per_airport[airport]:
                    new_days = np.random.randint(low=2,high=5)
                if index == len(airport_order) - 1:
                    days_per_airport[airport_order] += (new_days - days_per_airport[airport]) * -1
                else:
                    previous_change = (new_days - days_per_airport[airport]) * -1
                days_per_airport[airport] = new_days


        return ''.join([airport * days_per_airport.get(airport) for airport in days_per_airport.keys()])


    def calculate_fitness(self, individual: str) -> float:
        
        pass


    def select_parent(self, population: list[str]) -> str:
        
        pass
