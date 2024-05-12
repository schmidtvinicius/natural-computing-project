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

    def crossover(self, parent1: list[tuple[str, int]], parent2: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
        cities1 = [city for city, _ in parent1]
        cities2 = [city for city, _ in parent2]

        # Get the length of the tour
        tour_length = len(parent1)

        # Get positions to perform crossover
        pos1 = random.randint(0, tour_length - 1)
        pos2 = random.randint(0, tour_length - 1)

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        # generates first child
        child1 = cities1.copy()
        child1_parent1 = cities1[pos1:pos2] # get the cities from parent1 to be inserted in child1
        diff_parent2 = [city for city in cities2 if city not in child1_parent1] # get the cities from parent2 that are not in child1_parent1

        # insert the cities from parent2 that are not in child1_parent1
        idx_parent2 = 0
        for i in range(0,pos1):
            child1[i] = diff_parent2[idx_parent2]
            idx_parent2 += 1
        for i in range(pos2, tour_length):
            child1[i] = diff_parent2[idx_parent2]
            idx_parent2 += 1

        # generates second child
        child2 = cities2.copy()
        child2_parent2 = cities2[pos1:pos2] # get the cities from parent2 to be inserted in child2
        diff_parent1 = [city for city in cities1 if city not in child2_parent2] # get the cities from parent1 that are not in child2_parent2

        # insert the cities from parent1 that are not in child2_parent2
        idx_parent1 = 0
        for i in range(0,pos1):
            child2[i] = diff_parent1[idx_parent1]
            idx_parent1 += 1
        for i in range(pos2, tour_length):
            child2[i] = diff_parent1[idx_parent1]
            idx_parent1 += 1

        # Replace the cities with the cities and days
        for i in range(tour_length):
            child1[i] = (child1[i], parent1[i][1])
            child2[i] = (child2[i], parent2[i][1])

        return child1, child2

    def mutate(self, individual: list[tuple[str, int]]) -> list[tuple[str, int]]:
        min_day = 1
        max_day = individual[1][1]
        for i in range(len(individual)):
            # Get the current city and day
            city, day = individual[i]
            
            if random.random() < self.mutation_rate:
                # Generate a new day
                day = random.randint(min_day, max_day)
 
                # Update the city
                individual[i] = (city, day)

                print('mutation', individual[i])

            # Update the minimum and maximum day for the next city
            min_day = day
            if i < len(individual) - 1:
                max_day = individual[i+1][1]

        return individual

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
