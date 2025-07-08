"""
Differential Evolution implementation.
"""

import numpy as np
from typing import Tuple, List

class DifferentialEvolution:
    """Differential Evolution optimizer."""
    
    def __init__(self, 
                 problem,
                 population_size: int = 50,
                 generations: int = 100,
                 F: float = 0.5,
                 CR: float = 0.9):
        """
        Initialize Differential Evolution.
        
        Args:
            problem: Optimization problem instance
            population_size: Size of the population
            generations: Number of generations to run
            F: Scaling factor for mutation
            CR: Crossover probability
        """
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        
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
    
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run the differential evolution algorithm."""
        # Initialize population
        self.population = self.initialize_population()
        self.fitness_values = self.evaluate_population(self.population)
        
        fitness_history = []
        
        for generation in range(self.generations):
            # Record best fitness
            best_fitness = np.min(self.fitness_values)
            fitness_history.append(best_fitness)
            
            for i in range(self.population_size):
                # Select three random individuals (different from current)
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation: v = x_a + F * (x_b - x_c)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                
                # Ensure bounds
                mutant = np.clip(mutant, self.problem.bounds[0], self.problem.bounds[1])
                
                # Crossover
                trial = self.population[i].copy()
                random_j = np.random.randint(self.problem.dimensions)
                
                for j in range(self.problem.dimensions):
                    if np.random.random() < self.CR or j == random_j:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = self.problem.evaluate(trial)
                if trial_fitness < self.fitness_values[i]:
                    self.population[i] = trial
                    self.fitness_values[i] = trial_fitness
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}")
        
        # Return best solution
        best_idx = np.argmin(self.fitness_values)
        best_solution = self.population[best_idx]
        best_fitness = self.fitness_values[best_idx]
        
        return best_solution, best_fitness, fitness_history
