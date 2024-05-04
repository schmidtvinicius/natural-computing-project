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

            # Add the first day (start) and last day (return to initial airport)
            tour.insert(0, (airports[-1], 0))  # Start at the last airport
            tour.append((airports[-1], num_days))  # Return to the last airport

            population.append(tour)

        return population


    def crossover(self, parent1: list[tuple[str, int]], parent2: list[tuple[str, int]]) -> list[tuple[str, int]]:
        # Placeholder for crossover method
        pass

    def mutate(self, individual: list[tuple[str, int]]) -> list[tuple[str, int]]:
        # Placeholder for mutation method
        pass

    def calculate_fitness(self, individual: list[tuple[str, int]]) -> float:
        # Placeholder for fitness calculation method
        pass

    def select_parent(self, population: list[list[tuple[str, int]]]) -> list[tuple[str, int]]:
        # Placeholder for parent selection method
        pass