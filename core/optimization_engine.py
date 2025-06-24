#!/usr/bin/env python3
"""
Optimization Engine - Mathematical Optimization and Performance Tuning
====================================================================

This module implements a comprehensive optimization engine for Schwabot,
handling parameter optimization, strategy optimization, and performance tuning.

Core Mathematical Functions:
- Parameter Optimization: P* = argmin(L(P)) where L is loss function
- Strategy Optimization: S* = argmax(R(S)) where R is return function
- Performance Tuning: T* = argmin(C(T)) where C is cost function
- Multi-objective Optimization: F(x) = [f₁(x), f₂(x), ..., fₙ(x)]

Core Functionality:
- Mathematical optimization algorithms
- Parameter tuning and calibration
- Strategy performance optimization
- Multi-objective optimization
- Constraint handling and validation
- Optimization result analysis and reporting
"""

import logging
import json
import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import queue
import weakref
import traceback
import random
import math

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    PARAMETER = "parameter"
    STRATEGY = "strategy"
    PERFORMANCE = "performance"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINED = "constrained"

class OptimizationMethod(Enum):
    GRADIENT_DESCENT = "gradient_descent"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    SIMULATED_ANNEALING = "simulated_annealing"
    EVOLUTIONARY_STRATEGY = "evolutionary_strategy"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"

class OptimizationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class OptimizationParameter:
    name: str
    value: float
    min_value: float
    max_value: float
    step_size: float = 0.01
    parameter_type: str = "continuous"  # continuous, discrete, categorical
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationObjective:
    name: str
    function: Callable
    weight: float = 1.0
    target: Optional[float] = None
    minimize: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConstraint:
    name: str
    function: Callable
    constraint_type: str = "inequality"  # equality, inequality
    bound: float = 0.0
    tolerance: float = 1e-6
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationResult:
    optimization_id: str
    status: OptimizationStatus
    best_parameters: Dict[str, float]
    best_objective_value: float
    objective_values: List[float]
    parameter_history: List[Dict[str, float]]
    iterations: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    convergence_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class GradientDescentOptimizer:
    """Gradient descent optimization algorithm."""
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 tolerance: float = 1e-6, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.momentum = momentum
        self.velocity = None
    
    def optimize(self, objective_function: Callable, initial_params: Dict[str, float],
                param_bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
        """Run gradient descent optimization."""
        try:
            start_time = datetime.now()
            current_params = initial_params.copy()
            param_names = list(current_params.keys())
            
            # Initialize velocity
            self.velocity = {name: 0.0 for name in param_names}
            
            best_params = current_params.copy()
            best_value = objective_function(current_params)
            objective_values = [best_value]
            parameter_history = [current_params.copy()]
            convergence_history = []
            
            for iteration in range(self.max_iterations):
                # Calculate gradients (finite difference approximation)
                gradients = self._calculate_gradients(objective_function, current_params, param_bounds)
                
                # Update parameters with momentum
                for name in param_names:
                    # Update velocity
                    self.velocity[name] = (self.momentum * self.velocity[name] + 
                                         self.learning_rate * gradients[name])
                    
                    # Update parameter
                    new_value = current_params[name] - self.velocity[name]
                    
                    # Apply bounds
                    min_val, max_val = param_bounds[name]
                    new_value = max(min_val, min(max_val, new_value))
                    
                    current_params[name] = new_value
                
                # Evaluate objective
                current_value = objective_function(current_params)
                objective_values.append(current_value)
                parameter_history.append(current_params.copy())
                
                # Update best solution
                if current_value < best_value:
                    best_value = current_value
                    best_params = current_params.copy()
                
                # Check convergence
                convergence = abs(current_value - objective_values[-2]) if len(objective_values) > 1 else float('inf')
                convergence_history.append(convergence)
                
                if convergence < self.tolerance:
                    logger.info(f"Gradient descent converged after {iteration + 1} iterations")
                    break
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return OptimizationResult(
                optimization_id=f"gd_{int(time.time())}",
                status=OptimizationStatus.COMPLETED,
                best_parameters=best_params,
                best_objective_value=best_value,
                objective_values=objective_values,
                parameter_history=parameter_history,
                iterations=len(objective_values),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                convergence_history=convergence_history,
                metadata={'method': 'gradient_descent', 'learning_rate': self.learning_rate}
            )
            
        except Exception as e:
            logger.error(f"Error in gradient descent optimization: {e}")
            return OptimizationResult(
                optimization_id=f"gd_{int(time.time())}",
                status=OptimizationStatus.FAILED,
                best_parameters=initial_params,
                best_objective_value=float('inf'),
                objective_values=[],
                parameter_history=[],
                iterations=0,
                start_time=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def _calculate_gradients(self, objective_function: Callable, params: Dict[str, float],
                           param_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate gradients using finite differences."""
        try:
            gradients = {}
            epsilon = 1e-6
            
            for name, value in params.items():
                # Forward step
                params_forward = params.copy()
                params_forward[name] = value + epsilon
                f_forward = objective_function(params_forward)
                
                # Backward step
                params_backward = params.copy()
                params_backward[name] = value - epsilon
                f_backward = objective_function(params_backward)
                
                # Calculate gradient
                gradients[name] = (f_forward - f_backward) / (2 * epsilon)
            
            return gradients
            
        except Exception as e:
            logger.error(f"Error calculating gradients: {e}")
            return {name: 0.0 for name in params.keys()}

class GeneticAlgorithmOptimizer:
    """Genetic algorithm optimization."""
    
    def __init__(self, population_size: int = 50, generations: int = 100,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elite_size: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def optimize(self, objective_function: Callable, param_bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        try:
            start_time = datetime.now()
            param_names = list(param_bounds.keys())
            
            # Initialize population
            population = self._initialize_population(param_bounds)
            
            best_individual = None
            best_value = float('inf')
            objective_values = []
            parameter_history = []
            convergence_history = []
            
            for generation in range(self.generations):
                # Evaluate fitness
                fitness_scores = []
                for individual in population:
                    params = dict(zip(param_names, individual))
                    fitness = objective_function(params)
                    fitness_scores.append(fitness)
                    
                    if fitness < best_value:
                        best_value = fitness
                        best_individual = individual.copy()
                
                objective_values.append(best_value)
                parameter_history.append(dict(zip(param_names, best_individual)))
                
                # Calculate convergence
                if len(objective_values) > 1:
                    convergence = abs(objective_values[-1] - objective_values[-2])
                    convergence_history.append(convergence)
                
                # Selection and reproduction
                population = self._evolve_population(population, fitness_scores, param_bounds)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return OptimizationResult(
                optimization_id=f"ga_{int(time.time())}",
                status=OptimizationStatus.COMPLETED,
                best_parameters=dict(zip(param_names, best_individual)),
                best_objective_value=best_value,
                objective_values=objective_values,
                parameter_history=parameter_history,
                iterations=len(objective_values),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                convergence_history=convergence_history,
                metadata={'method': 'genetic_algorithm', 'population_size': self.population_size}
            )
            
        except Exception as e:
            logger.error(f"Error in genetic algorithm optimization: {e}")
            return OptimizationResult(
                optimization_id=f"ga_{int(time.time())}",
                status=OptimizationStatus.FAILED,
                best_parameters={},
                best_objective_value=float('inf'),
                objective_values=[],
                parameter_history=[],
                iterations=0,
                start_time=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def _initialize_population(self, param_bounds: Dict[str, Tuple[float, float]]) -> List[List[float]]:
        """Initialize random population."""
        population = []
        param_names = list(param_bounds.keys())
        
        for _ in range(self.population_size):
            individual = []
            for name in param_names:
                min_val, max_val = param_bounds[name]
                value = random.uniform(min_val, max_val)
                individual.append(value)
            population.append(individual)
        
        return population
    
    def _evolve_population(self, population: List[List[float]], fitness_scores: List[float],
                          param_bounds: Dict[str, Tuple[float, float]]) -> List[List[float]]:
        """Evolve population through selection, crossover, and mutation."""
        try:
            param_names = list(param_bounds.keys())
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                child1 = self._mutate(child1, param_bounds)
                child2 = self._mutate(child2, param_bounds)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            return new_population[:self.population_size]
            
        except Exception as e:
            logger.error(f"Error evolving population: {e}")
            return population
    
    def _tournament_selection(self, population: List[List[float]], 
                            fitness_scores: List[float]) -> List[float]:
        """Tournament selection."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Single-point crossover."""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def _mutate(self, individual: List[float], param_bounds: Dict[str, Tuple[float, float]]) -> List[float]:
        """Gaussian mutation."""
        param_names = list(param_bounds.keys())
        mutated = individual.copy()
        
        for i, name in enumerate(param_names):
            if random.random() < self.mutation_rate:
                min_val, max_val = param_bounds[name]
                std_dev = (max_val - min_val) * 0.1
                mutation = random.gauss(0, std_dev)
                mutated[i] = max(min_val, min(max_val, mutated[i] + mutation))
        
        return mutated

class ParticleSwarmOptimizer:
    """Particle swarm optimization."""
    
    def __init__(self, num_particles: int = 30, max_iterations: int = 100,
                 cognitive_weight: float = 2.0, social_weight: float = 2.0,
                 inertia_weight: float = 0.7):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_weight = inertia_weight
    
    def optimize(self, objective_function: Callable, param_bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
        """Run particle swarm optimization."""
        try:
            start_time = datetime.now()
            param_names = list(param_bounds.keys())
            num_dimensions = len(param_names)
            
            # Initialize particles
            particles = self._initialize_particles(param_bounds)
            velocities = self._initialize_velocities(param_bounds)
            
            # Initialize personal and global best
            personal_best = [particle.copy() for particle in particles]
            personal_best_values = [objective_function(dict(zip(param_names, particle))) for particle in particles]
            
            global_best_idx = np.argmin(personal_best_values)
            global_best = personal_best[global_best_idx].copy()
            global_best_value = personal_best_values[global_best_idx]
            
            objective_values = [global_best_value]
            parameter_history = [dict(zip(param_names, global_best))]
            convergence_history = []
            
            for iteration in range(self.max_iterations):
                for i in range(self.num_particles):
                    # Update velocity
                    for j in range(num_dimensions):
                        cognitive_component = (self.cognitive_weight * random.random() * 
                                             (personal_best[i][j] - particles[i][j]))
                        social_component = (self.social_weight * random.random() * 
                                          (global_best[j] - particles[i][j]))
                        
                        velocities[i][j] = (self.inertia_weight * velocities[i][j] + 
                                          cognitive_component + social_component)
                    
                    # Update position
                    for j in range(num_dimensions):
                        particles[i][j] += velocities[i][j]
                        
                        # Apply bounds
                        min_val, max_val = param_bounds[param_names[j]]
                        particles[i][j] = max(min_val, min(max_val, particles[i][j]))
                    
                    # Evaluate fitness
                    current_value = objective_function(dict(zip(param_names, particles[i])))
                    
                    # Update personal best
                    if current_value < personal_best_values[i]:
                        personal_best[i] = particles[i].copy()
                        personal_best_values[i] = current_value
                        
                        # Update global best
                        if current_value < global_best_value:
                            global_best = particles[i].copy()
                            global_best_value = current_value
                
                objective_values.append(global_best_value)
                parameter_history.append(dict(zip(param_names, global_best)))
                
                # Calculate convergence
                if len(objective_values) > 1:
                    convergence = abs(objective_values[-1] - objective_values[-2])
                    convergence_history.append(convergence)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return OptimizationResult(
                optimization_id=f"pso_{int(time.time())}",
                status=OptimizationStatus.COMPLETED,
                best_parameters=dict(zip(param_names, global_best)),
                best_objective_value=global_best_value,
                objective_values=objective_values,
                parameter_history=parameter_history,
                iterations=len(objective_values),
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                convergence_history=convergence_history,
                metadata={'method': 'particle_swarm', 'num_particles': self.num_particles}
            )
            
        except Exception as e:
            logger.error(f"Error in particle swarm optimization: {e}")
            return OptimizationResult(
                optimization_id=f"pso_{int(time.time())}",
                status=OptimizationStatus.FAILED,
                best_parameters={},
                best_objective_value=float('inf'),
                objective_values=[],
                parameter_history=[],
                iterations=0,
                start_time=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def _initialize_particles(self, param_bounds: Dict[str, Tuple[float, float]]) -> List[List[float]]:
        """Initialize particles randomly within bounds."""
        particles = []
        param_names = list(param_bounds.keys())
        
        for _ in range(self.num_particles):
            particle = []
            for name in param_names:
                min_val, max_val = param_bounds[name]
                value = random.uniform(min_val, max_val)
                particle.append(value)
            particles.append(particle)
        
        return particles
    
    def _initialize_velocities(self, param_bounds: Dict[str, Tuple[float, float]]) -> List[List[float]]:
        """Initialize velocities randomly."""
        velocities = []
        param_names = list(param_bounds.keys())
        
        for _ in range(self.num_particles):
            velocity = []
            for name in param_names:
                min_val, max_val = param_bounds[name]
                max_velocity = (max_val - min_val) * 0.1
                value = random.uniform(-max_velocity, max_velocity)
                velocity.append(value)
            velocities.append(velocity)
        
        return velocities

class OptimizationEngine:
    """Main optimization engine."""
    
    def __init__(self):
        self.optimizers: Dict[OptimizationMethod, Any] = {}
        self.optimization_history: deque = deque(maxlen=1000)
        self.current_optimizations: Dict[str, Any] = {}
        self._initialize_optimizers()
    
    def _initialize_optimizers(self) -> None:
        """Initialize available optimizers."""
        self.optimizers[OptimizationMethod.GRADIENT_DESCENT] = GradientDescentOptimizer()
        self.optimizers[OptimizationMethod.GENETIC_ALGORITHM] = GeneticAlgorithmOptimizer()
        self.optimizers[OptimizationMethod.PARTICLE_SWARM] = ParticleSwarmOptimizer()
    
    def optimize_parameters(self, objective_function: Callable, 
                          initial_params: Dict[str, float],
                          param_bounds: Dict[str, Tuple[float, float]],
                          method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT,
                          **kwargs) -> OptimizationResult:
        """Optimize parameters using specified method."""
        try:
            if method not in self.optimizers:
                logger.error(f"Optimization method {method.value} not available")
                return OptimizationResult(
                    optimization_id=f"opt_{int(time.time())}",
                    status=OptimizationStatus.FAILED,
                    best_parameters=initial_params,
                    best_objective_value=float('inf'),
                    objective_values=[],
                    parameter_history=[],
                    iterations=0,
                    start_time=datetime.now(),
                    metadata={'error': f"Method {method.value} not available"}
                )
            
            optimizer = self.optimizers[method]
            
            # Update optimizer parameters if provided
            for key, value in kwargs.items():
                if hasattr(optimizer, key):
                    setattr(optimizer, key, value)
            
            # Run optimization
            result = optimizer.optimize(objective_function, initial_params, param_bounds)
            
            # Store in history
            self.optimization_history.append(result)
            
            logger.info(f"Optimization completed: {result.optimization_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return OptimizationResult(
                optimization_id=f"opt_{int(time.time())}",
                status=OptimizationStatus.FAILED,
                best_parameters=initial_params,
                best_objective_value=float('inf'),
                objective_values=[],
                parameter_history=[],
                iterations=0,
                start_time=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def optimize_strategy(self, strategy_function: Callable,
                         param_bounds: Dict[str, Tuple[float, float]],
                         method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM,
                         **kwargs) -> OptimizationResult:
        """Optimize strategy parameters."""
        try:
            # Create objective function that maximizes strategy returns
            def objective_function(params):
                try:
                    return -strategy_function(params)  # Negative because we maximize returns
                except Exception as e:
                    logger.error(f"Error evaluating strategy: {e}")
                    return float('inf')
            
            return self.optimize_parameters(objective_function, {}, param_bounds, method, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in strategy optimization: {e}")
            return OptimizationResult(
                optimization_id=f"strategy_{int(time.time())}",
                status=OptimizationStatus.FAILED,
                best_parameters={},
                best_objective_value=float('inf'),
                objective_values=[],
                parameter_history=[],
                iterations=0,
                start_time=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def get_optimization_history(self, limit: int = 100) -> List[OptimizationResult]:
        """Get optimization history."""
        return list(self.optimization_history)[-limit:]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        try:
            if not self.optimization_history:
                return {'total_optimizations': 0}
            
            optimizations = list(self.optimization_history)
            
            # Calculate statistics
            completed = [opt for opt in optimizations if opt.status == OptimizationStatus.COMPLETED]
            failed = [opt for opt in optimizations if opt.status == OptimizationStatus.FAILED]
            
            if completed:
                best_values = [opt.best_objective_value for opt in completed]
                durations = [opt.duration for opt in completed if opt.duration]
                
                summary = {
                    'total_optimizations': len(optimizations),
                    'completed_optimizations': len(completed),
                    'failed_optimizations': len(failed),
                    'success_rate': len(completed) / len(optimizations),
                    'best_objective_value': min(best_values),
                    'avg_objective_value': np.mean(best_values),
                    'avg_duration': np.mean(durations) if durations else 0,
                    'methods_used': list(set(opt.metadata.get('method', 'unknown') for opt in optimizations))
                }
            else:
                summary = {
                    'total_optimizations': len(optimizations),
                    'completed_optimizations': 0,
                    'failed_optimizations': len(failed),
                    'success_rate': 0.0
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {'total_optimizations': 0, 'error': str(e)}

def main():
    """Main function for testing."""
    try:
        # Create optimization engine
        engine = OptimizationEngine()
        
        # Test objective function
        def test_objective(params):
            x = params.get('x', 0)
            y = params.get('y', 0)
            return (x - 2)**2 + (y - 3)**2  # Minimum at (2, 3)
        
        # Test parameter bounds
        param_bounds = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        initial_params = {'x': 0.0, 'y': 0.0}
        
        # Test different optimization methods
        methods = [
            OptimizationMethod.GRADIENT_DESCENT,
            OptimizationMethod.GENETIC_ALGORITHM,
            OptimizationMethod.PARTICLE_SWARM
        ]
        
        for method in methods:
            print(f"\nTesting {method.value}...")
            result = engine.optimize_parameters(
                test_objective, initial_params, param_bounds, method
            )
            
            print(f"Status: {result.status.value}")
            print(f"Best parameters: {result.best_parameters}")
            print(f"Best objective value: {result.best_objective_value:.6f}")
            print(f"Iterations: {result.iterations}")
            print(f"Duration: {result.duration:.2f}s")
        
        # Get optimization summary
        summary = engine.get_optimization_summary()
        print(f"\nOptimization Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 