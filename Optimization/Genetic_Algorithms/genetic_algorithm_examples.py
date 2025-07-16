"""
Genetic Algorithm Examples
==========================

- Basic genetic algorithm implementation
- Traveling salesman problem (TSP)
- Function optimization
- Parameter tuning
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable
import time

# 1. Basic Genetic Algorithm Implementation
print("=== Basic Genetic Algorithm Implementation ===")

class GeneticAlgorithm:
    def __init__(self, pop_size=50, mutation_rate=0.1, crossover_rate=0.8, elite_size=5):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
    def initialize_population(self, chromosome_length):
        """Initialize random population"""
        return np.random.randint(2, size=(self.pop_size, chromosome_length))
    
    def fitness_function(self, chromosome):
        """Simple fitness function: count number of 1s"""
        return np.sum(chromosome)
    
    def select_parents(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(2):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
            
        return selected[0], selected[1]
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutate(self, chromosome):
        """Bit-flip mutation"""
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def evolve(self, population, fitness_scores):
        """Evolve population for one generation"""
        new_population = []
        
        # Elitism: keep best individuals
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents(population, fitness_scores)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return np.array(new_population[:self.pop_size])
    
    def optimize(self, chromosome_length, generations=100):
        """Main optimization loop"""
        population = self.initialize_population(chromosome_length)
        best_fitness_history = []
        avg_fitness_history = []
        
        for generation in range(generations):
            # Calculate fitness
            fitness_scores = [self.fitness_function(chrom) for chrom in population]
            
            # Record statistics
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best = {best_fitness}, Avg = {avg_fitness:.2f}")
            
            # Check convergence
            if best_fitness == chromosome_length:
                print(f"Optimal solution found at generation {generation}")
                break
            
            # Evolve population
            population = self.evolve(population, fitness_scores)
        
        return population, best_fitness_history, avg_fitness_history

# Test basic GA
print("Testing basic genetic algorithm...")
ga = GeneticAlgorithm(pop_size=50, mutation_rate=0.1, crossover_rate=0.8)
population, best_history, avg_history = ga.optimize(chromosome_length=20, generations=100)

# 2. Traveling Salesman Problem (TSP)
print("\n=== Traveling Salesman Problem (TSP) ===")

class TSPGeneticAlgorithm:
    def __init__(self, pop_size=50, mutation_rate=0.1, crossover_rate=0.8, elite_size=5):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def create_random_route(self, num_cities):
        """Create a random route (permutation)"""
        return list(range(num_cities))
    
    def initialize_population(self, num_cities):
        """Initialize population with random routes"""
        population = []
        for _ in range(self.pop_size):
            route = self.create_random_route(num_cities)
            random.shuffle(route)
            population.append(route)
        return population
    
    def calculate_distance(self, route, distances):
        """Calculate total distance of a route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += distances[from_city][to_city]
        return total_distance
    
    def fitness_function(self, route, distances):
        """Fitness is inverse of distance"""
        distance = self.calculate_distance(route, distances)
        return 1.0 / (distance + 1e-10)  # Avoid division by zero
    
    def select_parents(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(2):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
            
        return selected[0], selected[1]
    
    def order_crossover(self, parent1, parent2):
        """Order crossover for TSP"""
        if random.random() < self.crossover_rate:
            size = len(parent1)
            start, end = sorted(random.sample(range(size), 2))
            
            # Create child1
            child1 = [-1] * size
            child1[start:end] = parent1[start:end]
            
            # Fill remaining positions with cities from parent2
            remaining = [city for city in parent2 if city not in child1[start:end]]
            j = 0
            for i in range(size):
                if child1[i] == -1:
                    child1[i] = remaining[j]
                    j += 1
            
            # Create child2
            child2 = [-1] * size
            child2[start:end] = parent2[start:end]
            
            remaining = [city for city in parent1 if city not in child2[start:end]]
            j = 0
            for i in range(size):
                if child2[i] == -1:
                    child2[i] = remaining[j]
                    j += 1
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def swap_mutation(self, route):
        """Swap mutation for TSP"""
        mutated = route.copy()
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def evolve(self, population, fitness_scores):
        """Evolve population for one generation"""
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents(population, fitness_scores)
            child1, child2 = self.order_crossover(parent1, parent2)
            child1 = self.swap_mutation(child1)
            child2 = self.swap_mutation(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.pop_size]
    
    def optimize(self, distances, generations=100):
        """Main optimization loop for TSP"""
        num_cities = len(distances)
        population = self.initialize_population(num_cities)
        best_distance_history = []
        avg_distance_history = []
        
        for generation in range(generations):
            # Calculate fitness
            fitness_scores = [self.fitness_function(route, distances) for route in population]
            distances_list = [self.calculate_distance(route, distances) for route in population]
            
            # Record statistics
            best_distance = min(distances_list)
            avg_distance = np.mean(distances_list)
            best_distance_history.append(best_distance)
            avg_distance_history.append(avg_distance)
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best = {best_distance:.2f}, Avg = {avg_distance:.2f}")
            
            # Evolve population
            population = self.evolve(population, fitness_scores)
        
        # Return best route
        final_fitness = [self.fitness_function(route, distances) for route in population]
        best_route_idx = np.argmax(final_fitness)
        best_route = population[best_route_idx]
        
        return best_route, best_distance_history, avg_distance_history

# Test TSP
print("Testing TSP with genetic algorithm...")
# Create random distance matrix
num_cities = 10
np.random.seed(42)
cities = np.random.rand(num_cities, 2) * 100
distances = np.zeros((num_cities, num_cities))

for i in range(num_cities):
    for j in range(num_cities):
        distances[i][j] = np.sqrt(np.sum((cities[i] - cities[j])**2))

tsp_ga = TSPGeneticAlgorithm(pop_size=50, mutation_rate=0.1, crossover_rate=0.8)
best_route, best_dist_history, avg_dist_history = tsp_ga.optimize(distances, generations=50)

print(f"Best route found: {best_route}")
print(f"Total distance: {tsp_ga.calculate_distance(best_route, distances):.2f}")

# 3. Function Optimization
print("\n=== Function Optimization ===")

class FunctionOptimizationGA:
    def __init__(self, pop_size=50, mutation_rate=0.1, crossover_rate=0.8, elite_size=5):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def encode_individual(self, x, bounds, bits_per_dim=16):
        """Encode continuous values to binary"""
        encoded = []
        for i, (val, (low, high)) in enumerate(zip(x, bounds)):
            # Normalize to [0, 1]
            normalized = (val - low) / (high - low)
            # Convert to binary
            binary = int(normalized * (2**bits_per_dim - 1))
            binary_str = format(binary, f'0{bits_per_dim}b')
            encoded.extend([int(bit) for bit in binary_str])
        return np.array(encoded)
    
    def decode_individual(self, binary, bounds, bits_per_dim=16):
        """Decode binary to continuous values"""
        decoded = []
        for i, (low, high) in enumerate(bounds):
            start_idx = i * bits_per_dim
            end_idx = start_idx + bits_per_dim
            binary_part = binary[start_idx:end_idx]
            
            # Convert binary to integer
            binary_str = ''.join(map(str, binary_part))
            integer_val = int(binary_str, 2)
            
            # Denormalize
            normalized = integer_val / (2**bits_per_dim - 1)
            value = low + normalized * (high - low)
            decoded.append(value)
        
        return np.array(decoded)
    
    def initialize_population(self, bounds, bits_per_dim=16):
        """Initialize population with random continuous values"""
        population = []
        for _ in range(self.pop_size):
            # Generate random continuous values
            individual = np.array([np.random.uniform(low, high) for low, high in bounds])
            # Encode to binary
            binary = self.encode_individual(individual, bounds, bits_per_dim)
            population.append(binary)
        return population
    
    def fitness_function(self, binary, objective_func, bounds, bits_per_dim=16):
        """Fitness is negative of objective function (for minimization)"""
        x = self.decode_individual(binary, bounds, bits_per_dim)
        return -objective_func(x)  # Negative because we maximize fitness
    
    def select_parents(self, population, fitness_scores):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(2):
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
            
        return selected[0], selected[1]
    
    def uniform_crossover(self, parent1, parent2):
        """Uniform crossover"""
        if random.random() < self.crossover_rate:
            mask = np.random.randint(2, size=len(parent1))
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def bit_flip_mutation(self, binary):
        """Bit-flip mutation"""
        mutated = binary.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def evolve(self, population, fitness_scores):
        """Evolve population for one generation"""
        new_population = []
        
        # Elitism
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Generate rest of population
        while len(new_population) < self.pop_size:
            parent1, parent2 = self.select_parents(population, fitness_scores)
            child1, child2 = self.uniform_crossover(parent1, parent2)
            child1 = self.bit_flip_mutation(child1)
            child2 = self.bit_flip_mutation(child2)
            
            new_population.extend([child1, child2])
        
        return np.array(new_population[:self.pop_size])
    
    def optimize(self, objective_func, bounds, generations=100, bits_per_dim=16):
        """Main optimization loop"""
        population = self.initialize_population(bounds, bits_per_dim)
        best_value_history = []
        avg_value_history = []
        
        for generation in range(generations):
            # Calculate fitness
            fitness_scores = [self.fitness_function(binary, objective_func, bounds, bits_per_dim) 
                            for binary in population]
            values = [-fitness for fitness in fitness_scores]  # Convert back to objective values
            
            # Record statistics
            best_value = min(values)
            avg_value = np.mean(values)
            best_value_history.append(best_value)
            avg_value_history.append(avg_value)
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best = {best_value:.6f}, Avg = {avg_value:.6f}")
            
            # Evolve population
            population = self.evolve(population, fitness_scores)
        
        # Return best solution
        final_fitness = [self.fitness_function(binary, objective_func, bounds, bits_per_dim) 
                        for binary in population]
        best_idx = np.argmax(final_fitness)
        best_binary = population[best_idx]
        best_solution = self.decode_individual(best_binary, bounds, bits_per_dim)
        
        return best_solution, best_value_history, avg_value_history

# Test function optimization
print("Testing function optimization...")
def rosenbrock_function(x):
    """Rosenbrock function for testing"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

bounds = [(-2, 2), (-2, 2)]
func_ga = FunctionOptimizationGA(pop_size=50, mutation_rate=0.1, crossover_rate=0.8)
best_solution, best_val_history, avg_val_history = func_ga.optimize(rosenbrock_function, bounds, generations=50)

print(f"Best solution found: {best_solution}")
print(f"Function value: {rosenbrock_function(best_solution):.6f}")

# 4. Parameter Tuning Example
print("\n=== Parameter Tuning Example ===")

def simple_classifier(x, params):
    """Simple classifier with parameters to tune"""
    # params[0]: threshold, params[1]: weight
    return 1 if x * params[1] > params[0] else 0

def evaluate_parameters(params, training_data):
    """Evaluate parameters on training data"""
    correct = 0
    total = len(training_data)
    
    for x, y_true in training_data:
        y_pred = simple_classifier(x, params)
        if y_pred == y_true:
            correct += 1
    
    return correct / total  # Accuracy

# Generate synthetic training data
np.random.seed(42)
training_data = []
for _ in range(100):
    x = np.random.uniform(-5, 5)
    # Simple rule: positive if x > 0
    y_true = 1 if x > 0 else 0
    training_data.append((x, y_true))

# Define parameter bounds
param_bounds = [(-5, 5), (0, 2)]  # threshold, weight

# Create GA for parameter tuning
param_ga = FunctionOptimizationGA(pop_size=30, mutation_rate=0.1, crossover_rate=0.8)

def objective_function(params):
    """Objective function for parameter tuning"""
    return -evaluate_parameters(params, training_data)  # Negative because we minimize

best_params, param_history, _ = param_ga.optimize(objective_function, param_bounds, generations=30)

print(f"Best parameters found: threshold={best_params[0]:.3f}, weight={best_params[1]:.3f}")
print(f"Best accuracy: {-objective_function(best_params):.3f}")

# 5. Visualization
print("\n=== Visualization ===")

plt.figure(figsize=(15, 10))

# Plot basic GA convergence
plt.subplot(2, 3, 1)
plt.plot(best_history, label='Best Fitness')
plt.plot(avg_history, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Basic GA Convergence')
plt.legend()
plt.grid(True)

# Plot TSP convergence
plt.subplot(2, 3, 2)
plt.plot(best_dist_history, label='Best Distance')
plt.plot(avg_dist_history, label='Average Distance')
plt.xlabel('Generation')
plt.ylabel('Distance')
plt.title('TSP GA Convergence')
plt.legend()
plt.grid(True)

# Plot TSP route
plt.subplot(2, 3, 3)
route_coords = [cities[i] for i in best_route] + [cities[best_route[0]]]
route_coords = np.array(route_coords)
plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-o')
plt.plot(cities[:, 0], cities[:, 1], 'ro', markersize=8)
plt.title('Best TSP Route')
plt.grid(True)

# Plot function optimization convergence
plt.subplot(2, 3, 4)
plt.plot(best_val_history, label='Best Value')
plt.plot(avg_val_history, label='Average Value')
plt.xlabel('Generation')
plt.ylabel('Function Value')
plt.title('Function Optimization Convergence')
plt.legend()
plt.grid(True)

# Plot Rosenbrock function contour
plt.subplot(2, 3, 5)
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosenbrock_function([X[i, j], Y[i, j]])

plt.contour(X, Y, Z, levels=20)
plt.plot(best_solution[0], best_solution[1], 'r*', markersize=10, label='Best Solution')
plt.plot(1, 1, 'g*', markersize=10, label='Global Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock Function')
plt.legend()
plt.grid(True)

# Plot parameter tuning convergence
plt.subplot(2, 3, 6)
plt.plot([-val for val in param_history], label='Best Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.title('Parameter Tuning Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("1. Basic GA: Successfully optimized binary string")
print("2. TSP GA: Found good route for traveling salesman problem")
print("3. Function Optimization: Optimized Rosenbrock function")
print("4. Parameter Tuning: Found optimal parameters for simple classifier") 