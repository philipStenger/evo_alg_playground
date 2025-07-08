# Evolutionary Algorithm Playground

A playground for experimenting with evolutionary algorithms and optimization techniques.

## Overview

This project provides implementations of various evolutionary algorithms including:
- Genetic Algorithm (GA)
- Differential Evolution (DE)
- Particle Swarm Optimization (PSO)
- Evolution Strategies (ES)
- Multi-objective optimization with NSGA-II

## Setup

1. Create a virtual environment:
```bash
python -m venv env
```

2. Activate the virtual environment:
```bash
# Windows
env\Scripts\activate
# Linux/Mac
source env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run different optimization algorithms:
```bash
# Genetic Algorithm
python main.py --algorithm ga --problem sphere

# Differential Evolution
python main.py --algorithm de --problem rastrigin

# Particle Swarm Optimization
python main.py --algorithm pso --problem rosenbrock
```

## Project Structure

- `src/` - Main source code
- `algorithms/` - Evolutionary algorithm implementations
- `problems/` - Optimization problem definitions
- `utils/` - Utility functions and visualization tools
- `experiments/` - Experiment configurations and results
