"""
Simulated Annealing Examples
============================

- Basic simulated annealing implementation
- TSP with simulated annealing
- Function optimization
- Scheduling problems
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# 1. Basic Simulated Annealing Implementation
print("=== Basic Simulated Annealing Implementation ===")

class SimulatedAnnealing:
    def __init__(self, initial_temp=100, cooling_rate=0.95, min_temp=0.1):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate acceptance probability"""
        if new_cost < current_cost:
            return 1.0  # Always accept better solutions
        else:
            # Accept worse solutions with some probability
            delta_e = new_cost - current_cost
            return math.exp(-delta_e / temperature)
    
    def optimize_binary_string(self, length=20, max_iterations=1000):
        """Optimize binary string to maximize number of 1s"""
        
        def objective_function(solution):
            return -np.sum(solution)  # Negative because we minimize
        
        def generate_neighbor(solution):
            """Generate neighbor by flipping one random bit"""
            neighbor = solution.copy()
            flip_idx = random.randint(0, len(neighbor) - 1)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            return neighbor
        
        # Initialize random solution
        current_solution = np.random.randint(2, size=length)
        current_cost = objective_function(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        iteration = 0
        history = []
        
        while temperature > self.min_temp and iteration < max_iterations:
            # Generate neighbor
            neighbor = generate_neighbor(current_solution)
            neighbor_cost = objective_function(neighbor)
            
            # Decide whether to accept neighbor
            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_solution = neighbor
                current_cost = neighbor_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            
            # Record history
            history.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_cost': current_cost,
                'best_cost': best_cost
            })
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Temp={temperature:.3f}, Best={-best_cost}")
        
        return best_solution, -best_cost, history

# Test basic SA
print("Testing basic simulated annealing...")
sa = SimulatedAnnealing(initial_temp=100, cooling_rate=0.95)
best_solution, best_fitness, history = sa.optimize_binary_string(length=20, max_iterations=500)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")

# 2. TSP with Simulated Annealing
print("\n=== TSP with Simulated Annealing ===")

class TSPSimulatedAnnealing:
    def __init__(self, initial_temp=100, cooling_rate=0.95, min_temp=0.1):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def calculate_distance(self, route, distances):
        """Calculate total distance of a route"""
        total_distance = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            total_distance += distances[from_city][to_city]
        return total_distance
    
    def generate_neighbor_2opt(self, route):
        """Generate neighbor using 2-opt swap"""
        neighbor = route.copy()
        i, j = sorted(random.sample(range(len(route)), 2))
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        return neighbor
    
    def generate_neighbor_swap(self, route):
        """Generate neighbor using random swap"""
        neighbor = route.copy()
        i, j = random.sample(range(len(route)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate acceptance probability"""
        if new_cost < current_cost:
            return 1.0
        else:
            delta_e = new_cost - current_cost
            return math.exp(-delta_e / temperature)
    
    def optimize(self, distances, max_iterations=1000):
        """Optimize TSP using simulated annealing"""
        num_cities = len(distances)
        
        # Initialize random route
        current_route = list(range(num_cities))
        random.shuffle(current_route)
        current_cost = self.calculate_distance(current_route, distances)
        
        best_route = current_route.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        iteration = 0
        history = []
        
        while temperature > self.min_temp and iteration < max_iterations:
            # Generate neighbor (alternate between 2-opt and swap)
            if random.random() < 0.5:
                neighbor = self.generate_neighbor_2opt(current_route)
            else:
                neighbor = self.generate_neighbor_swap(current_route)
            
            neighbor_cost = self.calculate_distance(neighbor, distances)
            
            # Decide whether to accept neighbor
            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_route = neighbor
                current_cost = neighbor_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_route = current_route.copy()
                    best_cost = current_cost
            
            # Record history
            history.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_cost': current_cost,
                'best_cost': best_cost
            })
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Temp={temperature:.3f}, Best={best_cost:.2f}")
        
        return best_route, best_cost, history

# Test TSP SA
print("Testing TSP with simulated annealing...")
# Create random distance matrix
num_cities = 10
np.random.seed(42)
cities = np.random.rand(num_cities, 2) * 100
distances = np.zeros((num_cities, num_cities))

for i in range(num_cities):
    for j in range(num_cities):
        distances[i][j] = np.sqrt(np.sum((cities[i] - cities[j])**2))

tsp_sa = TSPSimulatedAnnealing(initial_temp=100, cooling_rate=0.95)
best_route, best_distance, tsp_history = tsp_sa.optimize(distances, max_iterations=500)
print(f"Best route: {best_route}")
print(f"Best distance: {best_distance:.2f}")

# 3. Function Optimization
print("\n=== Function Optimization with Simulated Annealing ===")

class FunctionOptimizationSA:
    def __init__(self, initial_temp=100, cooling_rate=0.95, min_temp=0.1):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate acceptance probability"""
        if new_cost < current_cost:
            return 1.0
        else:
            delta_e = new_cost - current_cost
            return math.exp(-delta_e / temperature)
    
    def optimize(self, objective_func, bounds, max_iterations=1000, step_size=0.1):
        """Optimize continuous function using simulated annealing"""
        num_dimensions = len(bounds)
        
        # Initialize random solution
        current_solution = np.array([random.uniform(low, high) for low, high in bounds])
        current_cost = objective_func(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        iteration = 0
        history = []
        
        while temperature > self.min_temp and iteration < max_iterations:
            # Generate neighbor by adding random noise
            neighbor = current_solution + np.random.normal(0, step_size, num_dimensions)
            
            # Ensure neighbor is within bounds
            for i in range(num_dimensions):
                neighbor[i] = max(bounds[i][0], min(bounds[i][1], neighbor[i]))
            
            neighbor_cost = objective_func(neighbor)
            
            # Decide whether to accept neighbor
            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_solution = neighbor
                current_cost = neighbor_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            
            # Record history
            history.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_cost': current_cost,
                'best_cost': best_cost
            })
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Temp={temperature:.3f}, Best={best_cost:.6f}")
        
        return best_solution, best_cost, history

# Test function optimization
print("Testing function optimization...")
def rosenbrock_function(x):
    """Rosenbrock function for testing"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

bounds = [(-2, 2), (-2, 2)]
func_sa = FunctionOptimizationSA(initial_temp=100, cooling_rate=0.95)
best_solution, best_value, func_history = func_sa.optimize(rosenbrock_function, bounds, max_iterations=500)
print(f"Best solution: {best_solution}")
print(f"Best value: {best_value:.6f}")

# 4. Scheduling Problem
print("\n=== Scheduling Problem with Simulated Annealing ===")

class JobSchedulingSA:
    def __init__(self, initial_temp=100, cooling_rate=0.95, min_temp=0.1):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
    
    def calculate_makespan(self, schedule, processing_times):
        """Calculate makespan (total completion time)"""
        num_jobs = len(schedule)
        num_machines = len(processing_times[0])
        
        # Initialize machine completion times
        machine_times = [0] * num_machines
        
        # Process each job in the schedule
        for job_idx in schedule:
            for machine_idx in range(num_machines):
                if machine_idx == 0:
                    # First machine: add to previous completion time
                    machine_times[machine_idx] += processing_times[job_idx][machine_idx]
                else:
                    # Other machines: wait for previous machine and current machine
                    machine_times[machine_idx] = max(machine_times[machine_idx-1], machine_times[machine_idx]) + processing_times[job_idx][machine_idx]
        
        return machine_times[-1]  # Return completion time of last machine
    
    def generate_neighbor_swap(self, schedule):
        """Generate neighbor by swapping two jobs"""
        neighbor = schedule.copy()
        i, j = random.sample(range(len(schedule)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor
    
    def generate_neighbor_insert(self, schedule):
        """Generate neighbor by inserting a job at a different position"""
        neighbor = schedule.copy()
        if len(neighbor) > 1:
            i, j = random.sample(range(len(schedule)), 2)
            job = neighbor.pop(i)
            neighbor.insert(j, job)
        return neighbor
    
    def acceptance_probability(self, current_cost, new_cost, temperature):
        """Calculate acceptance probability"""
        if new_cost < current_cost:
            return 1.0
        else:
            delta_e = new_cost - current_cost
            return math.exp(-delta_e / temperature)
    
    def optimize(self, processing_times, max_iterations=1000):
        """Optimize job scheduling using simulated annealing"""
        num_jobs = len(processing_times)
        
        # Initialize random schedule
        current_schedule = list(range(num_jobs))
        random.shuffle(current_schedule)
        current_cost = self.calculate_makespan(current_schedule, processing_times)
        
        best_schedule = current_schedule.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        iteration = 0
        history = []
        
        while temperature > self.min_temp and iteration < max_iterations:
            # Generate neighbor (alternate between swap and insert)
            if random.random() < 0.5:
                neighbor = self.generate_neighbor_swap(current_schedule)
            else:
                neighbor = self.generate_neighbor_insert(current_schedule)
            
            neighbor_cost = self.calculate_makespan(neighbor, processing_times)
            
            # Decide whether to accept neighbor
            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_schedule = neighbor
                current_cost = neighbor_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_schedule = current_schedule.copy()
                    best_cost = current_cost
            
            # Record history
            history.append({
                'iteration': iteration,
                'temperature': temperature,
                'current_cost': current_cost,
                'best_cost': best_cost
            })
            
            # Cool down
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Temp={temperature:.3f}, Best={best_cost}")
        
        return best_schedule, best_cost, history

# Test job scheduling
print("Testing job scheduling...")
# Create random processing times (jobs x machines)
num_jobs = 8
num_machines = 3
np.random.seed(42)
processing_times = np.random.randint(1, 10, size=(num_jobs, num_machines))

scheduling_sa = JobSchedulingSA(initial_temp=100, cooling_rate=0.95)
best_schedule, best_makespan, scheduling_history = scheduling_sa.optimize(processing_times, max_iterations=500)
print(f"Best schedule: {best_schedule}")
print(f"Best makespan: {best_makespan}")

# 5. Visualization
print("\n=== Visualization ===")

plt.figure(figsize=(15, 10))

# Plot basic SA convergence
plt.subplot(2, 3, 1)
iterations = [h['iteration'] for h in history]
best_costs = [-h['best_cost'] for h in history]  # Convert back to positive
temperatures = [h['temperature'] for h in history]

plt.plot(iterations, best_costs, label='Best Fitness')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title('Basic SA Convergence')
plt.legend()
plt.grid(True)

# Plot TSP SA convergence
plt.subplot(2, 3, 2)
tsp_iterations = [h['iteration'] for h in tsp_history]
tsp_best_costs = [h['best_cost'] for h in tsp_history]
plt.plot(tsp_iterations, tsp_best_costs, label='Best Distance')
plt.xlabel('Iteration')
plt.ylabel('Distance')
plt.title('TSP SA Convergence')
plt.legend()
plt.grid(True)

# Plot TSP route
plt.subplot(2, 3, 3)
route_coords = [cities[i] for i in best_route] + [cities[best_route[0]]]
route_coords = np.array(route_coords)
plt.plot(route_coords[:, 0], route_coords[:, 1], 'b-o')
plt.plot(cities[:, 0], cities[:, 1], 'ro', markersize=8)
plt.title('Best TSP Route (SA)')
plt.grid(True)

# Plot function optimization convergence
plt.subplot(2, 3, 4)
func_iterations = [h['iteration'] for h in func_history]
func_best_costs = [h['best_cost'] for h in func_history]
plt.plot(func_iterations, func_best_costs, label='Best Value')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Function Optimization SA Convergence')
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
plt.title('Rosenbrock Function (SA)')
plt.legend()
plt.grid(True)

# Plot scheduling convergence
plt.subplot(2, 3, 6)
sched_iterations = [h['iteration'] for h in scheduling_history]
sched_best_costs = [h['best_cost'] for h in scheduling_history]
plt.plot(sched_iterations, sched_best_costs, label='Best Makespan')
plt.xlabel('Iteration')
plt.ylabel('Makespan')
plt.title('Job Scheduling SA Convergence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Temperature scheduling comparison
print("\n=== Temperature Scheduling Comparison ===")

def exponential_cooling(initial_temp, iteration, alpha=0.95):
    """Exponential cooling schedule"""
    return initial_temp * (alpha ** iteration)

def linear_cooling(initial_temp, iteration, max_iterations):
    """Linear cooling schedule"""
    return initial_temp * (1 - iteration / max_iterations)

def logarithmic_cooling(initial_temp, iteration, max_iterations):
    """Logarithmic cooling schedule"""
    return initial_temp / (1 + iteration)

# Test different cooling schedules
max_iterations = 500
initial_temp = 100

# Generate cooling schedules
iterations = range(max_iterations)
exp_temp = [exponential_cooling(initial_temp, i) for i in iterations]
lin_temp = [linear_cooling(initial_temp, i, max_iterations) for i in iterations]
log_temp = [logarithmic_cooling(initial_temp, i, max_iterations) for i in iterations]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(iterations, exp_temp, label='Exponential', linewidth=2)
plt.plot(iterations, lin_temp, label='Linear', linewidth=2)
plt.plot(iterations, log_temp, label='Logarithmic', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Temperature')
plt.title('Cooling Schedules')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(iterations, exp_temp, label='Exponential', linewidth=2)
plt.semilogy(iterations, lin_temp, label='Linear', linewidth=2)
plt.semilogy(iterations, log_temp, label='Logarithmic', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Temperature (log scale)')
plt.title('Cooling Schedules (Log Scale)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== Summary ===")
print("1. Basic SA: Successfully optimized binary string")
print("2. TSP SA: Found good route for traveling salesman problem")
print("3. Function Optimization SA: Optimized Rosenbrock function")
print("4. Job Scheduling SA: Found optimal job sequence")
print("5. Different cooling schedules compared") 