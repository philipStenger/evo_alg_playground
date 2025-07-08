"""
Standard test functions for optimization benchmarking.
"""

import numpy as np

class OptimizationProblem:
    """Base class for optimization problems."""
    
    def __init__(self, dimensions: int, bounds: tuple):
        self.dimensions = dimensions
        self.bounds = bounds
        
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function."""
        raise NotImplementedError

class SphereFunction(OptimizationProblem):
    """Sphere function: f(x) = sum(x_i^2)"""
    
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-5.12, 5.12))
        
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)

class RastriginFunction(OptimizationProblem):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-5.12, 5.12))
        self.A = 10
        
    def evaluate(self, x: np.ndarray) -> float:
        return (self.A * len(x) + 
                np.sum(x**2 - self.A * np.cos(2 * np.pi * x)))

class RosenbrockFunction(OptimizationProblem):
    """Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-2.048, 2.048))
        
    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

class AckleyFunction(OptimizationProblem):
    """Ackley function"""
    
    def __init__(self, dimensions: int = 10):
        super().__init__(dimensions, (-32.768, 32.768))
        
    def evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) -
                np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

def get_problem(problem_name: str, dimensions: int = 10) -> OptimizationProblem:
    """Factory function to get optimization problems."""
    problems = {
        'sphere': SphereFunction,
        'rastrigin': RastriginFunction,
        'rosenbrock': RosenbrockFunction,
        'ackley': AckleyFunction
    }
    
    if problem_name not in problems:
        raise ValueError(f"Unknown problem: {problem_name}")
    
    return problems[problem_name](dimensions)
