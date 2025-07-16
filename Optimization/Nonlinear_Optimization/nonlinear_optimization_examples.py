"""
Nonlinear Optimization Examples
==============================

- Gradient Descent Methods (detailed)
- Newton's Method (basic)
- BFGS Optimization (basic)
- Constrained Optimization (basic)
- Function optimization examples
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, BFGS
import time

# 1. Detailed Gradient Descent Methods
print("=== Detailed Gradient Descent Methods ===")

def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def gradient_descent_basic(f, grad_f, x0, learning_rate=0.001, max_iter=1000, tol=1e-6):
    """Basic gradient descent implementation"""
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, history

def gradient_descent_momentum(f, grad_f, x0, learning_rate=0.001, momentum=0.9, max_iter=1000, tol=1e-6):
    """Gradient descent with momentum"""
    x = np.array(x0, dtype=float)
    velocity = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        velocity = momentum * velocity - learning_rate * gradient
        x_new = x + velocity
        
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, history

def gradient_descent_adagrad(f, grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6):
    """Gradient descent with AdaGrad"""
    x = np.array(x0, dtype=float)
    squared_grad_sum = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        squared_grad_sum += gradient**2
        adaptive_lr = learning_rate / (np.sqrt(squared_grad_sum) + 1e-8)
        x_new = x - adaptive_lr * gradient
        
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, history

def gradient_descent_rmsprop(f, grad_f, x0, learning_rate=0.01, beta=0.9, max_iter=1000, tol=1e-6):
    """Gradient descent with RMSprop"""
    x = np.array(x0, dtype=float)
    squared_grad_avg = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        squared_grad_avg = beta * squared_grad_avg + (1 - beta) * gradient**2
        adaptive_lr = learning_rate / (np.sqrt(squared_grad_avg) + 1e-8)
        x_new = x - adaptive_lr * gradient
        
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, history

def gradient_descent_adam(f, grad_f, x0, learning_rate=0.01, beta1=0.9, beta2=0.999, max_iter=1000, tol=1e-6):
    """Gradient descent with Adam"""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)  # First moment
    v = np.zeros_like(x)  # Second moment
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * gradient
        
        # Update biased second moment estimate
        v = beta2 * v + (1 - beta2) * gradient**2
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**(i+1))
        
        # Compute bias-corrected second moment estimate
        v_hat = v / (1 - beta2**(i+1))
        
        # Update parameters
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, history

# Test different gradient descent methods
x0 = np.array([-1.5, -1.5])
print(f"Starting point: {x0}")
print(f"True minimum: [1, 1]")

methods = [
    ("Basic GD", gradient_descent_basic, {"learning_rate": 0.001}),
    ("Momentum GD", gradient_descent_momentum, {"learning_rate": 0.001, "momentum": 0.9}),
    ("AdaGrad", gradient_descent_adagrad, {"learning_rate": 0.1}),
    ("RMSprop", gradient_descent_rmsprop, {"learning_rate": 0.01}),
    ("Adam", gradient_descent_adam, {"learning_rate": 0.01})
]

results = {}
for name, method, params in methods:
    print(f"\n--- {name} ---")
    start_time = time.time()
    x_opt, history = method(rosenbrock, rosenbrock_gradient, x0, **params)
    end_time = time.time()
    
    results[name] = {
        'x_opt': x_opt,
        'history': history,
        'time': end_time - start_time,
        'iterations': len(history)
    }
    
    print(f"Optimal point: {x_opt}")
    print(f"Function value: {rosenbrock(x_opt):.6f}")
    print(f"Time: {end_time - start_time:.4f} seconds")
    print(f"Iterations: {len(history)}")

# 2. Newton's Method (Basic)
print("\n=== Newton's Method (Basic) ===")

def newton_method_1d(f, df, d2f, x0, max_iter=100, tol=1e-6):
    """Newton's method for 1D optimization"""
    x = x0
    history = [x]
    
    for i in range(max_iter):
        x_new = x - df(x) / d2f(x)
        
        if abs(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        x = x_new
        history.append(x)
    
    return x, history

# Example: minimize f(x) = x^2 + 2x + 1
def f(x): return x**2 + 2*x + 1
def df(x): return 2*x + 2
def d2f(x): return 2

x0_newton = 5.0
x_opt_newton, history_newton = newton_method_1d(f, df, d2f, x0_newton)
print(f"Newton's method result: x = {x_opt_newton:.6f}, f(x) = {f(x_opt_newton):.6f}")

# 3. BFGS Optimization (Basic)
print("\n=== BFGS Optimization (Basic) ===")

# Using scipy's BFGS
x0_bfgs = np.array([-1.5, -1.5])
result_bfgs = minimize(rosenbrock, x0_bfgs, method='BFGS', jac=rosenbrock_gradient)
print(f"BFGS result: {result_bfgs.x}")
print(f"Function value: {result_bfgs.fun:.6f}")
print(f"Success: {result_bfgs.success}")
print(f"Iterations: {result_bfgs.nit}")

# 4. Constrained Optimization (Basic)
print("\n=== Constrained Optimization (Basic) ===")

def objective(x):
    """Objective function: f(x,y) = x^2 + y^2"""
    return x[0]**2 + x[1]**2

def constraint(x):
    """Constraint: x + y >= 1"""
    return x[0] + x[1] - 1

# Using scipy's SLSQP for constrained optimization
x0_constrained = np.array([0.5, 0.5])
constraints = {'type': 'ineq', 'fun': constraint}
result_constrained = minimize(objective, x0_constrained, method='SLSQP', constraints=constraints)

print(f"Constrained optimization result: {result_constrained.x}")
print(f"Function value: {result_constrained.fun:.6f}")
print(f"Constraint value: {constraint(result_constrained.x):.6f}")

# 5. Visualization of optimization paths
print("\n=== Visualization ===")

# Create contour plot
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosenbrock([X[i, j], Y[i, j]])

plt.figure(figsize=(15, 10))

# Plot optimization paths
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (name, data) in enumerate(results.items()):
    history = np.array(data['history'])
    plt.subplot(2, 3, i+1)
    plt.contour(X, Y, Z, levels=20)
    plt.plot(history[:, 0], history[:, 1], 'o-', color=colors[i], label=name, markersize=3)
    plt.plot(1, 1, 'k*', markersize=10, label='Global minimum')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{name} Optimization Path')
    plt.legend()
    plt.grid(True)

# Plot convergence comparison
plt.subplot(2, 3, 6)
for name, data in results.items():
    history = np.array(data['history'])
    function_values = [rosenbrock(x) for x in history]
    plt.semilogy(function_values, label=name, linewidth=2)

plt.xlabel('Iteration')
plt.ylabel('Function Value (log scale)')
plt.title('Convergence Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. Performance comparison
print("\n=== Performance Comparison ===")
print(f"{'Method':<15} {'Iterations':<12} {'Time (s)':<10} {'Final Value':<12}")
print("-" * 50)
for name, data in results.items():
    print(f"{name:<15} {data['iterations']:<12} {data['time']:<10.4f} {rosenbrock(data['x_opt']):<12.6f}")

# 7. Additional function examples
print("\n=== Additional Function Examples ===")

def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2)"""
    return np.sum(x**2)

def sphere_gradient(x):
    """Gradient of sphere function"""
    return 2 * x

def rastrigin_function(x):
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))"""
    A = 10
    n = len(x)
    return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_gradient(x):
    """Gradient of Rastrigin function"""
    A = 10
    return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)

# Test on different functions
test_functions = [
    ("Sphere", sphere_function, sphere_gradient, np.array([2.0, 2.0])),
    ("Rastrigin", rastrigin_function, rastrigin_gradient, np.array([2.0, 2.0]))
]

for name, func, grad_func, start_point in test_functions:
    print(f"\n--- {name} Function ---")
    print(f"Starting point: {start_point}")
    
    # Use Adam for comparison
    x_opt, _ = gradient_descent_adam(func, grad_func, start_point, learning_rate=0.01)
    print(f"Optimal point: {x_opt}")
    print(f"Function value: {func(x_opt):.6f}") 