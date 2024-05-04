import pandas as pd
import random

from ga.ga import GeneticAlgorithm

class GeneticAlgorithmRLE(GeneticAlgorithm):
    def __init__(
        self, 
        population_size: int,
        mutation_rate: float,
        crossover_rate: float,
        num_generations: int,
        dataset: pd.DataFrame
    ):
        super().__init__(population_size, mutation_rate, crossover_rate, num_generations, dataset)

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
                # Select a random day for each airport
                day = random.randint(current_day + 1, num_days)
                tour.append((airport, day))
                current_day = day

            # Add the first day (start) and last day (return to initial airport)
            tour.insert(0, (airports[-1], 0))  # Start at the last airport
            tour.append((airports[-1], num_days))  # Return to the last airport

            population.append(tour)

        return population
