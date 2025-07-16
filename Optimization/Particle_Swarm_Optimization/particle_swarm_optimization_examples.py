"""
Particle Swarm Optimization Examples
====================================

- Basic PSO implementation
- Function optimization
- Neural network weight optimization
- Multi-objective optimization (detailed)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random

# 1. Basic Particle Swarm Optimization Implementation
print("=== Basic Particle Swarm Optimization Implementation ===")

class ParticleSwarmOptimization:
    def __init__(self, n_particles=30, n_dimensions=2, w=0.7, c1=1.5, c2=1.5, max_iter=100):
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive learning factor
        self.c2 = c2  # Social learning factor
        self.max_iter = max_iter
        
        # Initialize particles
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_values = None
        self.gbest_position = None
        self.gbest_value = float('inf')
        
    def initialize_particles(self, bounds):
        """Initialize particle positions and velocities"""
        self.positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            (self.n_particles, self.n_dimensions)
        )
        
        # Initialize velocities (small random values)
        velocity_range = bounds[:, 1] - bounds[:, 0]
        self.velocities = np.random.uniform(
            -velocity_range * 0.1, velocity_range * 0.1,
            (self.n_particles, self.n_dimensions)
        )
        
        # Initialize personal best
        self.pbest_positions = self.positions.copy()
        self.pbest_values = np.full(self.n_particles, float('inf'))
        
    def update_particles(self, objective_function, bounds):
        """Update particle positions and velocities"""
        for i in range(self.n_particles):
            # Evaluate current position
            current_value = objective_function(self.positions[i])
            
            # Update personal best
            if current_value < self.pbest_values[i]:
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_values[i] = current_value
                
                # Update global best
                if current_value < self.gbest_value:
                    self.gbest_position = self.positions[i].copy()
                    self.gbest_value = current_value
            
            # Update velocity
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (self.gbest_position - self.positions[i])
            
            self.velocities[i] = (self.w * self.velocities[i] + 
                                cognitive_velocity + social_velocity)
            
            # Update position
            self.positions[i] += self.velocities[i]
            
            # Clamp to bounds
            self.positions[i] = np.clip(self.positions[i], bounds[:, 0], bounds[:, 1])
    
    def optimize(self, objective_function, bounds):
        """Main optimization loop"""
        self.initialize_particles(bounds)
        history = []
        
        for iteration in range(self.max_iter):
            self.update_particles(objective_function, bounds)
            
            # Record history
            history.append({
                'iteration': iteration,
                'gbest_value': self.gbest_value,
                'gbest_position': self.gbest_position.copy(),
                'avg_value': np.mean([objective_function(pos) for pos in self.positions])
            })
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best = {self.gbest_value:.6f}")
        
        return self.gbest_position, self.gbest_value, history

# Test basic PSO
print("Testing basic PSO...")
def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2)

bounds = np.array([[-5, 5], [-5, 5]])
pso = ParticleSwarmOptimization(n_particles=30, max_iter=100)
best_position, best_value, history = pso.optimize(sphere_function, bounds)
print(f"Best position: {best_position}")
print(f"Best value: {best_value:.6f}")

# 2. Function Optimization with PSO
print("\n=== Function Optimization with PSO ===")

def rosenbrock_function(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rastrigin_function(x):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley_function(x):
    """Ackley function"""
    n = len(x)
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n)) - 
            np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)

# Test different functions
test_functions = [
    ("Sphere", sphere_function, np.array([[-5, 5], [-5, 5]])),
    ("Rosenbrock", rosenbrock_function, np.array([[-2, 2], [-2, 2]])),
    ("Rastrigin", rastrigin_function, np.array([[-5.12, 5.12], [-5.12, 5.12]])),
    ("Ackley", ackley_function, np.array([[-5, 5], [-5, 5]]))
]

function_results = {}
for name, func, bounds in test_functions:
    print(f"\n--- {name} Function ---")
    pso_func = ParticleSwarmOptimization(n_particles=30, max_iter=100)
    best_pos, best_val, hist = pso_func.optimize(func, bounds)
    function_results[name] = {'position': best_pos, 'value': best_val, 'history': hist}
    print(f"Best position: {best_pos}")
    print(f"Best value: {best_val:.6f}")

# 3. Neural Network Weight Optimization with PSO
print("\n=== Neural Network Weight Optimization with PSO ===")

class PSONeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, n_particles=50, max_iter=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_particles = n_particles
        self.max_iter = max_iter
        
        # Calculate total number of weights
        self.n_weights = (input_size * hidden_size + hidden_size +  # W1, b1
                         hidden_size * output_size + output_size)    # W2, b2
        
    def weights_to_matrices(self, weights):
        """Convert flat weight array to neural network matrices"""
        idx = 0
        
        # W1: input_size x hidden_size
        w1_size = self.input_size * self.hidden_size
        W1 = weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # b1: hidden_size
        b1 = weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        # W2: hidden_size x output_size
        w2_size = self.hidden_size * self.output_size
        W2 = weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # b2: output_size
        b2 = weights[idx:idx + self.output_size]
        
        return W1, b1, W2, b2
    
    def forward_pass(self, X, weights):
        """Forward pass through the neural network"""
        W1, b1, W2, b2 = self.weights_to_matrices(weights)
        
        # Hidden layer
        hidden = np.tanh(X @ W1 + b1)
        
        # Output layer
        output = hidden @ W2 + b2
        
        return output
    
    def objective_function(self, weights, X_train, y_train):
        """Objective function: mean squared error"""
        predictions = self.forward_pass(X_train, weights)
        return mean_squared_error(y_train, predictions)
    
    def optimize_weights(self, X_train, y_train, X_test, y_test):
        """Optimize neural network weights using PSO"""
        # Initialize PSO
        bounds = np.array([[-2, 2]] * self.n_weights)
        
        def objective(w):
            return self.objective_function(w, X_train, y_train)
        
        pso = ParticleSwarmOptimization(n_particles=self.n_particles, max_iter=self.max_iter)
        best_weights, best_mse, history = pso.optimize(objective, bounds)
        
        # Evaluate on test set
        test_predictions = self.forward_pass(X_test, best_weights)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        return best_weights, best_mse, test_mse, history

# Test PSO neural network optimization
print("Testing PSO neural network optimization...")
# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=5, n_targets=1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

pso_nn = PSONeuralNetwork(input_size=5, hidden_size=10, output_size=1, n_particles=30, max_iter=50)
best_weights, train_mse, test_mse, nn_history = pso_nn.optimize_weights(X_train, y_train, X_test, y_test)

print(f"Training MSE: {train_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")

# 4. Multi-Objective Optimization (Detailed)
print("\n=== Multi-Objective Optimization (Detailed) ===")

class MultiObjectivePSO:
    def __init__(self, n_particles=50, n_objectives=2, w=0.7, c1=1.5, c2=1.5, max_iter=100):
        self.n_particles = n_particles
        self.n_objectives = n_objectives
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        
        # Initialize particles
        self.positions = None
        self.velocities = None
        self.pbest_positions = None
        self.pbest_values = None
        self.archive = []  # Pareto front archive
        
    def dominates(self, obj1, obj2):
        """Check if obj1 dominates obj2 (minimization)"""
        return (np.all(obj1 <= obj2) and np.any(obj1 < obj2))
    
    def is_non_dominated(self, objectives, archive):
        """Check if solution is non-dominated"""
        for archive_obj in archive:
            if self.dominates(archive_obj, objectives):
                return False
        return True
    
    def update_archive(self, positions, objectives_list):
        """Update Pareto front archive"""
        for i, objectives in enumerate(objectives_list):
            # Remove dominated solutions from archive
            self.archive = [obj for obj in self.archive 
                          if not self.dominates(objectives, obj)]
            
            # Add current solution if non-dominated
            if self.is_non_dominated(objectives, self.archive):
                self.archive.append(objectives)
    
    def select_leader(self):
        """Select leader from archive using crowding distance"""
        if len(self.archive) == 0:
            return None
        
        if len(self.archive) == 1:
            return self.archive[0]
        
        # Simple random selection from archive
        return random.choice(self.archive)
    
    def initialize_particles(self, bounds):
        """Initialize particle positions and velocities"""
        self.positions = np.random.uniform(
            bounds[:, 0], bounds[:, 1], 
            (self.n_particles, bounds.shape[0])
        )
        
        velocity_range = bounds[:, 1] - bounds[:, 0]
        self.velocities = np.random.uniform(
            -velocity_range * 0.1, velocity_range * 0.1,
            (self.n_particles, bounds.shape[0])
        )
        
        self.pbest_positions = self.positions.copy()
        self.pbest_values = np.full((self.n_particles, self.n_objectives), float('inf'))
    
    def update_particles(self, objective_functions, bounds):
        """Update particle positions and velocities"""
        objectives_list = []
        
        for i in range(self.n_particles):
            # Evaluate current position
            current_objectives = np.array([func(self.positions[i]) for func in objective_functions])
            objectives_list.append(current_objectives)
            
            # Update personal best (Pareto dominance)
            if self.dominates(current_objectives, self.pbest_values[i]):
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_values[i] = current_objectives.copy()
            
            # Select leader from archive
            leader = self.select_leader()
            if leader is None:
                leader = self.pbest_values[i]
            
            # Update velocity
            r1, r2 = np.random.rand(2)
            cognitive_velocity = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
            social_velocity = self.c2 * r2 * (leader - self.positions[i])
            
            self.velocities[i] = (self.w * self.velocities[i] + 
                                cognitive_velocity + social_velocity)
            
            # Update position
            self.positions[i] += self.velocities[i]
            
            # Clamp to bounds
            self.positions[i] = np.clip(self.positions[i], bounds[:, 0], bounds[:, 1])
        
        # Update archive
        self.update_archive(self.positions, objectives_list)
    
    def optimize(self, objective_functions, bounds):
        """Main multi-objective optimization loop"""
        self.initialize_particles(bounds)
        history = []
        
        for iteration in range(self.max_iter):
            self.update_particles(objective_functions, bounds)
            
            # Record history
            history.append({
                'iteration': iteration,
                'archive_size': len(self.archive),
                'archive': self.archive.copy() if self.archive else []
            })
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Archive size = {len(self.archive)}")
        
        return self.archive, history

# Test multi-objective optimization
print("Testing multi-objective PSO...")

def objective1(x):
    """First objective: minimize x^2 + y^2"""
    return x[0]**2 + x[1]**2

def objective2(x):
    """Second objective: minimize (x-1)^2 + (y-1)^2"""
    return (x[0] - 1)**2 + (x[1] - 1)**2

def objective3(x):
    """Third objective: minimize (x+1)^2 + (y+1)^2"""
    return (x[0] + 1)**2 + (x[1] + 1)**2

# Two-objective problem
print("Two-objective optimization:")
mopso_2d = MultiObjectivePSO(n_particles=50, n_objectives=2, max_iter=100)
bounds_2d = np.array([[-3, 3], [-3, 3]])
archive_2d, history_2d = mopso_2d.optimize([objective1, objective2], bounds_2d)

print(f"Pareto front size: {len(archive_2d)}")
for i, solution in enumerate(archive_2d[:5]):  # Show first 5 solutions
    print(f"Solution {i+1}: f1={solution[0]:.4f}, f2={solution[1]:.4f}")

# Three-objective problem
print("\nThree-objective optimization:")
mopso_3d = MultiObjectivePSO(n_particles=50, n_objectives=3, max_iter=100)
bounds_3d = np.array([[-3, 3], [-3, 3]])
archive_3d, history_3d = mopso_3d.optimize([objective1, objective2, objective3], bounds_3d)

print(f"Pareto front size: {len(archive_3d)}")
for i, solution in enumerate(archive_3d[:5]):  # Show first 5 solutions
    print(f"Solution {i+1}: f1={solution[0]:.4f}, f2={solution[1]:.4f}, f3={solution[2]:.4f}")

# 5. Visualization
print("\n=== Visualization ===")

plt.figure(figsize=(15, 10))

# Plot basic PSO convergence
plt.subplot(2, 3, 1)
iterations = [h['iteration'] for h in history]
gbest_values = [h['gbest_value'] for h in history]
avg_values = [h['avg_value'] for h in history]

plt.plot(iterations, gbest_values, label='Best Value', linewidth=2)
plt.plot(iterations, avg_values, label='Average Value', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Basic PSO Convergence')
plt.legend()
plt.grid(True)

# Plot function optimization results
plt.subplot(2, 3, 2)
function_names = list(function_results.keys())
best_values = [function_results[name]['value'] for name in function_names]
plt.bar(function_names, best_values)
plt.ylabel('Best Function Value')
plt.title('PSO Performance on Different Functions')
plt.xticks(rotation=45)

# Plot neural network optimization
plt.subplot(2, 3, 3)
nn_iterations = [h['iteration'] for h in nn_history]
nn_gbest_values = [h['gbest_value'] for h in nn_history]
plt.plot(nn_iterations, nn_gbest_values, label='Training MSE', linewidth=2)
plt.axhline(y=test_mse, color='r', linestyle='--', label=f'Test MSE: {test_mse:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.title('Neural Network Weight Optimization')
plt.legend()
plt.grid(True)

# Plot two-objective Pareto front
plt.subplot(2, 3, 4)
if archive_2d:
    archive_array = np.array(archive_2d)
    plt.scatter(archive_array[:, 0], archive_array[:, 1], c='red', s=50, alpha=0.7)
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Two-Objective Pareto Front')
    plt.grid(True)

# Plot three-objective Pareto front (first two objectives)
plt.subplot(2, 3, 5)
if archive_3d:
    archive_array = np.array(archive_3d)
    scatter = plt.scatter(archive_array[:, 0], archive_array[:, 1], 
                         c=archive_array[:, 2], cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Objective 3')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Three-Objective Pareto Front')
    plt.grid(True)

# Plot archive size evolution
plt.subplot(2, 3, 6)
iterations_2d = [h['iteration'] for h in history_2d]
archive_sizes_2d = [h['archive_size'] for h in history_2d]
iterations_3d = [h['iteration'] for h in history_3d]
archive_sizes_3d = [h['archive_size'] for h in history_3d]

plt.plot(iterations_2d, archive_sizes_2d, label='2 Objectives', linewidth=2)
plt.plot(iterations_3d, archive_sizes_3d, label='3 Objectives', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Archive Size')
plt.title('Pareto Front Evolution')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Performance comparison
print("\n=== Performance Comparison ===")
print("Function Optimization Results:")
print(f"{'Function':<12} {'Best Value':<12} {'Best Position'}")
print("-" * 40)
for name, result in function_results.items():
    print(f"{name:<12} {result['value']:<12.6f} {result['position']}")

print(f"\nNeural Network Results:")
print(f"Training MSE: {train_mse:.6f}")
print(f"Test MSE: {test_mse:.6f}")

print(f"\nMulti-Objective Results:")
print(f"Two-objective Pareto front size: {len(archive_2d)}")
print(f"Three-objective Pareto front size: {len(archive_3d)}")

print("\n=== Summary ===")
print("1. Basic PSO: Successfully optimized sphere function")
print("2. Function Optimization: Tested on multiple benchmark functions")
print("3. Neural Network Optimization: Optimized NN weights using PSO")
print("4. Multi-Objective Optimization: Found Pareto-optimal solutions")
print("5. Comprehensive visualization of all results") 