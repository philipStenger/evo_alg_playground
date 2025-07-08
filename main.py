"""
Evolutionary Algorithm Playground

Main entry point for running evolutionary optimization experiments.
"""

import argparse
import numpy as np
from pathlib import Path

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.differential_evolution import DifferentialEvolution
from src.algorithms.particle_swarm import ParticleSwarmOptimization
from src.problems.test_functions import get_problem
from src.utils.visualization import plot_convergence, plot_population
from src.utils.logging import setup_logger

def main():
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm Playground')
    parser.add_argument('--algorithm', type=str, default='ga',
                       choices=['ga', 'de', 'pso', 'es'],
                       help='Evolutionary algorithm to use')
    parser.add_argument('--problem', type=str, default='sphere',
                       choices=['sphere', 'rastrigin', 'rosenbrock', 'ackley'],
                       help='Optimization problem to solve')
    parser.add_argument('--dimensions', type=int, default=10,
                       help='Problem dimensionality')
    parser.add_argument('--population_size', type=int, default=50,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of generations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logger('evo_algorithms')
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Get optimization problem
    problem = get_problem(args.problem, args.dimensions)
    logger.info(f"Solving {args.problem} problem in {args.dimensions} dimensions")
    
    # Initialize algorithm
    if args.algorithm == 'ga':
        algorithm = GeneticAlgorithm(
            problem=problem,
            population_size=args.population_size,
            generations=args.generations
        )
    elif args.algorithm == 'de':
        algorithm = DifferentialEvolution(
            problem=problem,
            population_size=args.population_size,
            generations=args.generations
        )
    elif args.algorithm == 'pso':
        algorithm = ParticleSwarmOptimization(
            problem=problem,
            population_size=args.population_size,
            generations=args.generations
        )
    else:
        raise NotImplementedError(f"Algorithm {args.algorithm} not implemented yet")
    
    logger.info(f"Running {args.algorithm.upper()} with population size {args.population_size}")
    
    # Run optimization
    best_solution, best_fitness, fitness_history = algorithm.run()
    
    # Results
    logger.info(f"Best fitness: {best_fitness:.6f}")
    logger.info(f"Best solution: {best_solution}")
    
    # Visualization
    if args.visualize:
        logger.info("Creating visualizations...")
        
        # Create results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Plot convergence
        plot_convergence(fitness_history, 
                        save_path=results_dir / f"{args.algorithm}_{args.problem}_convergence.png")
        
        # Plot population (for 2D problems)
        if args.dimensions == 2:
            plot_population(algorithm.population, problem,
                          save_path=results_dir / f"{args.algorithm}_{args.problem}_population.png")
        
        logger.info(f"Plots saved to {results_dir}")

if __name__ == "__main__":
    main()
