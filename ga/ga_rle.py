from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import random

from ga import GeneticAlgorithm

class GeneticAlgorithmRLE(GeneticAlgorithm):
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

    def initialize_population(self) -> list[list[tuple[str, int]]]:
        population = []
        num_days = len(self.dataset['flightDate'].unique())
        airports = self.dataset['startingAirport'].unique()

        for _ in range(self.population_size):
            tour = []
            current_day = 0

            # Randomly shuffle airports to determine the order of visit
            random.shuffle(airports)

            for airport in airports:
                # Select a random day for each airport within the valid range
                day = random.randint(current_day + 1, num_days) if current_day < num_days else current_day
                tour.append((airport, day))
                current_day = day

            population.append(tour)

        return population


    def crossover(self, parent1: list[tuple[str, int]], parent2: list[tuple[str, int]]) -> list[tuple[str, int]]:
        # Placeholder for crossover method
        pass

    def mutate(self, individual: list[tuple[str, int]]) -> list[tuple[str, int]]:
        # Placeholder for mutation method
        pass

    def calculate_fitness(self, individual: list[tuple[str, int]]) -> float:
        total_cost = 0
        first_day = self.dataset['flightDate'].min()

        for i in range(1, len(individual)):
            start_city, start_day = individual[i-1]
            end_city, end_day = individual[i]

            # Check if the traveler stays less than one day in a city (except for the last city)
            if i < len(individual) - 2 and end_day - start_day <= 0:
                print('error 1')
                return np.inf

            # Check if a flight from start_city to end_city on end_day exists
            flight_exists = self.dataset[
                (self.dataset['startingAirport'] == start_city) &
                (self.dataset['destinationAirport'] == end_city) &
                (self.dataset['flightDate'] == (datetime.strptime(first_day, '%Y-%m-%d') + timedelta(days=start_day)).strftime('%Y-%m-%d'))
            ].shape[0] > 0

            if not flight_exists:
                return np.inf

            # Get the price of the flight
            flight_price = self.dataset[
                (self.dataset['startingAirport'] == start_city) &
                (self.dataset['destinationAirport'] == end_city) &
                (self.dataset['flightDate'] == (datetime.strptime(first_day, '%Y-%m-%d') + timedelta(days=start_day)).strftime('%Y-%m-%d'))
            ]['totalFare'].values[0]

            total_cost += flight_price

        return total_cost

    def select_parent(self, population: list[list[tuple[str, int]]]) -> list[tuple[str, int]]:
        # Placeholder for parent selection method
        pass
