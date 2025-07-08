"""
Visualization utilities for evolutionary algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from pathlib import Path

def plot_convergence(fitness_history: List[float], 
                    title: str = "Convergence Plot",
                    save_path: Optional[Path] = None):
    """Plot convergence of fitness over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, 'b-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_population(population: np.ndarray, 
                   problem,
                   title: str = "Population Distribution",
                   save_path: Optional[Path] = None):
    """Plot population distribution for 2D problems."""
    if population.shape[1] != 2:
        print("Population plotting only supported for 2D problems")
        return
    
    # Create meshgrid for contour plot
    x_range = np.linspace(problem.bounds[0], problem.bounds[1], 100)
    y_range = np.linspace(problem.bounds[0], problem.bounds[1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = problem.evaluate(np.array([X[i, j], Y[i, j]]))
    
    plt.figure(figsize=(10, 8))
    
    # Plot contours
    contours = plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.contourf(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
    plt.colorbar(label='Fitness')
    
    # Plot population
    plt.scatter(population[:, 0], population[:, 1], 
               c='red', s=50, alpha=0.7, edgecolors='black')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
