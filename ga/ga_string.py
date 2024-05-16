import datetime
import pandas as pd
import numpy as np
import random
import re

from collections import Counter
from datetime import datetime, timedelta
from ga import GeneticAlgorithm

class GeneticAlgorithmString(GeneticAlgorithm):
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        num_generations: int,
        dataset: pd.DataFrame,
        elitism: bool = True,
        individual_length: int = 3,
        seed: int = 42
    ):
        super().__init__(
            population_size,
            mutation_rate,
            num_generations,
            dataset,
            elitism,
            seed
        )
        self.individual_length = individual_length
    

    def count_airport_frequencies(self, individual: str) -> Counter:
        return Counter(re.findall('[A-Z]{'+ str(self.individual_length) +'}', individual))
    

    def initialize_population(self) -> list[str]:
        population = []
        total_days = len(self.dataset['flightDate'].unique())
        airports = self.dataset['startingAirport'].unique()

        for _ in np.arange(self.population_size):
            airport_order = np.copy(airports)
            np.random.shuffle(airport_order)
            days = np.random.randint(low=2,high=5,size=airports.size)
            while days.sum() != total_days:
                days = np.random.randint(low=2,high=5,size=airports.size)
            population.append(''.join([airport * day for day, airport in zip(airport_order, days)]))
        
        return population


    def crossover(self, parent1: str, parent2: str) -> list[str]:
        if len(parent1) != len(parent2):
            raise RuntimeError('Parents should have the same length')
    

        days_parent1 = self.count_airport_frequencies(parent1)
        days_parent2 = self.count_airport_frequencies(parent2)
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
        days_per_airport = self.count_airport_frequencies(individual)
        airports = list(days_per_airport.keys())

        for airport in days_per_airport.keys():
            if np.random.rand() < self.mutation_rate:
                swap_airport = np.random.choice(airports)
                while swap_airport == airport:
                    swap_airport = np.random.choice(airports)
                days_per_airport[airport], days_per_airport[swap_airport] = days_per_airport[swap_airport], days_per_airport[airport]

        return ''.join([airport * days_per_airport.get(airport) for airport in days_per_airport.keys()])


    def calculate_fitness(self, individual: str) -> float:
        total_cost = 0
        current_day = self.dataset['flightDate'].min()
        days_per_airport = list(self.count_airport_frequencies(individual).items())

        for i, (departure_city, days_spent) in enumerate(days_per_airport):
            departure_day = (datetime.strptime(current_day, '%Y-%m-%d') + timedelta(days=days_spent)).strftime('%Y-%m-%d')

            if i < len(days_per_airport) - 1:
                next_city, _ = days_per_airport[i + 1]
            else: break

            # Check if a flight from start_city to next_city on next_days exists
            flight_exists = self.dataset[
                (self.dataset['startingAirport'] == departure_city) &
                (self.dataset['destinationAirport'] == next_city) &
                (self.dataset['flightDate'] == departure_day)
            ].shape[0] > 0

            if not flight_exists:
                return np.inf

            # Get the price of the flight
            flight_price = self.dataset[
                (self.dataset['startingAirport'] == departure_city) &
                (self.dataset['destinationAirport'] == next_city) &
                (self.dataset['flightDate'] == departure_day)
            ]['totalFare'].values[0]

            current_day = departure_day
            total_cost += flight_price

        return total_cost
