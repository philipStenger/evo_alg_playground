"""
Genetic Algorithm implementation for continuous optimization.
"""

import numpy as np
from typing import Tuple, List, Callable

class GeneticAlgorithm:
    """Genetic Algorithm optimizer."""
    
    def __init__(self, 
                 problem,
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problem: Optimization problem instance
            population_size: Size of the population
            generations: Number of generations to run
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            tournament_size: Size of tournament for selection
        """
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        self.population = None
        self.fitness_values = None
        
    def initialize_population(self) -> np.ndarray:
        """Initialize random population."""
        return np.random.uniform(
            self.problem.bounds[0], 
            self.problem.bounds[1], 
            (self.population_size, self.problem.dimensions)
        )
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of entire population."""
        return np.array([self.problem.evaluate(individual) for individual in population])
    
    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection for parent selection."""
        selected = []
        for _ in range(self.population_size):
            # Select random individuals for tournament
            tournament_indices = np.random.choice(
                self.population_size, self.tournament_size, replace=False
            )
            # Find best individual in tournament
            best_idx = tournament_indices[np.argmin(fitness[tournament_indices])]
            selected.append(population[best_idx].copy())
        return np.array(selected)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta = 20  # Distribution index
        offspring1 = np.zeros_like(parent1)
        offspring2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    
                    # Calculate beta
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                    
                    # Generate offspring
                    offspring1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    offspring2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                else:
                    offspring1[i] = parent1[i]
                    offspring2[i] = parent2[i]
            else:
                offspring1[i] = parent1[i]
                offspring2[i] = parent2[i]
        
        # Ensure bounds
        offspring1 = np.clip(offspring1, self.problem.bounds[0], self.problem.bounds[1])
        offspring2 = np.clip(offspring2, self.problem.bounds[0], self.problem.bounds[1])
        
        return offspring1, offspring2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        if np.random.random() > self.mutation_rate:
            return individual.copy()
        
        eta = 20  # Distribution index
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() <= (1.0 / len(individual)):
                y = individual[i]
                delta1 = (y - self.problem.bounds[0]) / (self.problem.bounds[1] - self.problem.bounds[0])
                delta2 = (self.problem.bounds[1] - y) / (self.problem.bounds[1] - self.problem.bounds[0])
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                    deltaq = 1.0 - val ** mut_pow
                
                y = y + deltaq * (self.problem.bounds[1] - self.problem.bounds[0])
                mutated[i] = np.clip(y, self.problem.bounds[0], self.problem.bounds[1])
        
        return mutated
    
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run the genetic algorithm."""
        # Initialize population
        self.population = self.initialize_population()
        self.fitness_values = self.evaluate_population(self.population)
        
        fitness_history = []
        
        for generation in range(self.generations):
            # Record best fitness
            best_fitness = np.min(self.fitness_values)
            fitness_history.append(best_fitness)
            
            # Selection
            parents = self.tournament_selection(self.population, self.fitness_values)
            
            # Create offspring
            offspring = []
            for i in range(0, self.population_size - 1, 2):
                parent1, parent2 = parents[i], parents[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])
            
            # Ensure we have exactly population_size offspring
            offspring = offspring[:self.population_size]
            self.population = np.array(offspring)
            self.fitness_values = self.evaluate_population(self.population)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        # Return best solution
        best_idx = np.argmin(self.fitness_values)
        best_solution = self.population[best_idx]
        best_fitness = self.fitness_values[best_idx]
        
        return best_solution, best_fitness, fitness_history
