"""
Particle Swarm Optimization implementation.
"""

import numpy as np
from typing import Tuple, List

class ParticleSwarmOptimization:
    """Particle Swarm Optimization algorithm."""
    
    def __init__(self, 
                 problem,
                 population_size: int = 50,
                 generations: int = 100,
                 w: float = 0.729,
                 c1: float = 1.494,
                 c2: float = 1.494):
        """
        Initialize Particle Swarm Optimization.
        
        Args:
            problem: Optimization problem instance
            population_size: Size of the swarm (number of particles)
            generations: Number of iterations to run
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.population = None  # Current positions
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
        
    def initialize_swarm(self):
        """Initialize swarm positions and velocities."""
        # Initialize positions
        self.population = np.random.uniform(
            self.problem.bounds[0], 
            self.problem.bounds[1], 
            (self.population_size, self.problem.dimensions)
        )
        
        # Initialize velocities
        velocity_range = (self.problem.bounds[1] - self.problem.bounds[0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range, velocity_range,
            (self.population_size, self.problem.dimensions)
        )
        
        # Initialize personal bests
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.array([
            self.problem.evaluate(particle) for particle in self.population
        ])
        
        # Initialize global best
        best_idx = np.argmin(self.personal_best_fitness)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
    
    def update_velocities(self):
        """Update particle velocities."""
        r1 = np.random.random((self.population_size, self.problem.dimensions))
        r2 = np.random.random((self.population_size, self.problem.dimensions))
        
        cognitive_component = self.c1 * r1 * (self.personal_best - self.population)
        social_component = self.c2 * r2 * (self.global_best - self.population)
        
        self.velocities = (self.w * self.velocities + 
                          cognitive_component + social_component)
    
    def update_positions(self):
        """Update particle positions."""
        self.population += self.velocities
        
        # Ensure bounds
        self.population = np.clip(
            self.population, self.problem.bounds[0], self.problem.bounds[1]
        )
    
    def update_personal_bests(self):
        """Update personal best positions."""
        current_fitness = np.array([
            self.problem.evaluate(particle) for particle in self.population
        ])
        
        # Update personal bests where current fitness is better
        better_mask = current_fitness < self.personal_best_fitness
        self.personal_best[better_mask] = self.population[better_mask]
        self.personal_best_fitness[better_mask] = current_fitness[better_mask]
        
        # Update global best
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
    
    def run(self) -> Tuple[np.ndarray, float, List[float]]:
        """Run the particle swarm optimization algorithm."""
        # Initialize swarm
        self.initialize_swarm()
        
        fitness_history = []
        
        for generation in range(self.generations):
            # Record best fitness
            fitness_history.append(self.global_best_fitness)
            
            # Update velocities and positions
            self.update_velocities()
            self.update_positions()
            
            # Update personal and global bests
            self.update_personal_bests()
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.global_best_fitness:.6f}")
        
        return self.global_best, self.global_best_fitness, fitness_history
