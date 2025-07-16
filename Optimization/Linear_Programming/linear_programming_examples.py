"""
Linear Programming Examples
==========================

- Linear Programming (LP) with PuLP
- Mixed Integer Linear Programming (MILP)
- Transportation problem
- Assignment problem
- Resource allocation
- Production planning
"""

import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Basic Linear Programming Problem
print("=== Basic Linear Programming Problem ===")
print("Maximize: 3x + 4y")
print("Subject to:")
print("  x + 2y <= 8")
print("  3x + y <= 9")
print("  x, y >= 0")

# Create the model
prob = pulp.LpProblem("Basic_LP", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable("x", 0, None)
y = pulp.LpVariable("y", 0, None)

# Objective function
prob += 3*x + 4*y

# Constraints
prob += x + 2*y <= 8
prob += 3*x + y <= 9

# Solve the problem
prob.solve()

print(f"\nStatus: {pulp.LpStatus[prob.status]}")
print(f"Optimal value: {pulp.value(prob.objective):.2f}")
print(f"x = {pulp.value(x):.2f}")
print(f"y = {pulp.value(y):.2f}")

# 2. Mixed Integer Linear Programming (MILP)
print("\n=== Mixed Integer Linear Programming ===")
print("Maximize: 2x + 3y")
print("Subject to:")
print("  x + y <= 5")
print("  2x + y <= 8")
print("  x, y >= 0")
print("  x must be integer")

# Create MILP model
milp_prob = pulp.LpProblem("MILP_Example", pulp.LpMaximize)

# Decision variables (x is integer, y is continuous)
x_int = pulp.LpVariable("x", 0, None, cat='Integer')
y_cont = pulp.LpVariable("y", 0, None)

# Objective function
milp_prob += 2*x_int + 3*y_cont

# Constraints
milp_prob += x_int + y_cont <= 5
milp_prob += 2*x_int + y_cont <= 8

# Solve
milp_prob.solve()

print(f"\nStatus: {pulp.LpStatus[milp_prob.status]}")
print(f"Optimal value: {pulp.value(milp_prob.objective):.2f}")
print(f"x (integer) = {pulp.value(x_int):.0f}")
print(f"y (continuous) = {pulp.value(y_cont):.2f}")

# 3. Transportation Problem
print("\n=== Transportation Problem ===")
print("Minimize total transportation cost")

# Supply and demand data
supply = {'Factory_A': 100, 'Factory_B': 150, 'Factory_C': 200}
demand = {'Warehouse_1': 120, 'Warehouse_2': 80, 'Warehouse_3': 150, 'Warehouse_4': 100}

# Transportation costs
costs = {
    ('Factory_A', 'Warehouse_1'): 10, ('Factory_A', 'Warehouse_2'): 15,
    ('Factory_A', 'Warehouse_3'): 20, ('Factory_A', 'Warehouse_4'): 25,
    ('Factory_B', 'Warehouse_1'): 12, ('Factory_B', 'Warehouse_2'): 8,
    ('Factory_B', 'Warehouse_3'): 18, ('Factory_B', 'Warehouse_4'): 22,
    ('Factory_C', 'Warehouse_1'): 16, ('Factory_C', 'Warehouse_2'): 14,
    ('Factory_C', 'Warehouse_3'): 12, ('Factory_C', 'Warehouse_4'): 10
}

# Create transportation model
trans_prob = pulp.LpProblem("Transportation", pulp.LpMinimize)

# Decision variables
routes = [(f, w) for f in supply for w in demand]
route_vars = pulp.LpVariable.dicts("Route", routes, 0, None)

# Objective function: minimize total cost
trans_prob += pulp.lpSum([route_vars[f, w] * costs[f, w] for f, w in routes])

# Supply constraints
for f in supply:
    trans_prob += pulp.lpSum([route_vars[f, w] for w in demand]) <= supply[f]

# Demand constraints
for w in demand:
    trans_prob += pulp.lpSum([route_vars[f, w] for f in supply]) >= demand[w]

# Solve
trans_prob.solve()

print(f"\nStatus: {pulp.LpStatus[trans_prob.status]}")
print(f"Total cost: ${pulp.value(trans_prob.objective):.2f}")

# Display solution
print("\nOptimal transportation plan:")
for f, w in routes:
    if pulp.value(route_vars[f, w]) > 0:
        print(f"{f} -> {w}: {pulp.value(route_vars[f, w]):.0f} units")

# 4. Assignment Problem
print("\n=== Assignment Problem ===")
print("Assign workers to tasks to minimize total cost")

# Cost matrix (worker x task)
cost_matrix = np.array([
    [9, 2, 7, 8],
    [6, 4, 3, 7],
    [5, 8, 1, 8],
    [7, 6, 9, 4]
])

workers = ['Worker_1', 'Worker_2', 'Worker_3', 'Worker_4']
tasks = ['Task_1', 'Task_2', 'Task_3', 'Task_4']

# Create assignment model
assign_prob = pulp.LpProblem("Assignment", pulp.LpMinimize)

# Binary decision variables
assign_vars = pulp.LpVariable.dicts("Assign", 
                                   [(w, t) for w in workers for t in tasks], 
                                   0, 1, cat='Binary')

# Objective function
assign_prob += pulp.lpSum([assign_vars[w, t] * cost_matrix[i, j] 
                          for i, w in enumerate(workers) 
                          for j, t in enumerate(tasks)])

# Each worker assigned to exactly one task
for w in workers:
    assign_prob += pulp.lpSum([assign_vars[w, t] for t in tasks]) == 1

# Each task assigned to exactly one worker
for t in tasks:
    assign_prob += pulp.lpSum([assign_vars[w, t] for w in workers]) == 1

# Solve
assign_prob.solve()

print(f"\nStatus: {pulp.LpStatus[assign_prob.status]}")
print(f"Total cost: {pulp.value(assign_prob.objective):.0f}")

print("\nOptimal assignment:")
for w in workers:
    for t in tasks:
        if pulp.value(assign_vars[w, t]) == 1:
            print(f"{w} -> {t}")

# 5. Resource Allocation Problem
print("\n=== Resource Allocation Problem ===")
print("Allocate budget to maximize total return")

# Projects and their returns
projects = {
    'Project_A': {'return': 0.15, 'cost': 100000},
    'Project_B': {'return': 0.12, 'cost': 80000},
    'Project_C': {'return': 0.18, 'cost': 120000},
    'Project_D': {'return': 0.10, 'cost': 60000},
    'Project_E': {'return': 0.20, 'cost': 150000}
}

total_budget = 300000

# Create resource allocation model
alloc_prob = pulp.LpProblem("Resource_Allocation", pulp.LpMaximize)

# Binary decision variables (invest or not)
invest_vars = pulp.LpVariable.dicts("Invest", projects.keys(), 0, 1, cat='Binary')

# Objective function: maximize total return
alloc_prob += pulp.lpSum([invest_vars[p] * projects[p]['return'] * projects[p]['cost'] 
                         for p in projects])

# Budget constraint
alloc_prob += pulp.lpSum([invest_vars[p] * projects[p]['cost'] for p in projects]) <= total_budget

# Solve
alloc_prob.solve()

print(f"\nStatus: {pulp.LpStatus[alloc_prob.status]}")
print(f"Total return: ${pulp.value(alloc_prob.objective):.2f}")

print("\nInvestment plan:")
total_invested = 0
for p in projects:
    if pulp.value(invest_vars[p]) == 1:
        cost = projects[p]['cost']
        total_invested += cost
        print(f"{p}: ${cost:,} (Return: {projects[p]['return']*100:.1f}%)")

print(f"\nTotal invested: ${total_invested:,}")
print(f"Remaining budget: ${total_budget - total_invested:,}")

# 6. Production Planning Problem
print("\n=== Production Planning Problem ===")
print("Plan production to maximize profit")

# Products and their data
products = {
    'Product_1': {'profit': 50, 'labor': 2, 'material': 3, 'capacity': 100},
    'Product_2': {'profit': 80, 'labor': 3, 'material': 4, 'capacity': 80},
    'Product_3': {'profit': 120, 'labor': 4, 'material': 5, 'capacity': 60}
}

# Available resources
available_labor = 400
available_material = 500

# Create production planning model
prod_prob = pulp.LpProblem("Production_Planning", pulp.LpMaximize)

# Decision variables (production quantities)
prod_vars = pulp.LpVariable.dicts("Produce", products.keys(), 0, None)

# Objective function: maximize total profit
prod_prob += pulp.lpSum([prod_vars[p] * products[p]['profit'] for p in products])

# Labor constraint
prod_prob += pulp.lpSum([prod_vars[p] * products[p]['labor'] for p in products]) <= available_labor

# Material constraint
prod_prob += pulp.lpSum([prod_vars[p] * products[p]['material'] for p in products]) <= available_material

# Capacity constraints
for p in products:
    prod_prob += prod_vars[p] <= products[p]['capacity']

# Solve
prod_prob.solve()

print(f"\nStatus: {pulp.LpStatus[prod_prob.status]}")
print(f"Total profit: ${pulp.value(prod_prob.objective):.2f}")

print("\nProduction plan:")
for p in products:
    quantity = pulp.value(prod_vars[p])
    if quantity > 0:
        profit = quantity * products[p]['profit']
        print(f"{p}: {quantity:.0f} units (Profit: ${profit:.2f})")

# Resource usage
total_labor = sum(pulp.value(prod_vars[p]) * products[p]['labor'] for p in products)
total_material = sum(pulp.value(prod_vars[p]) * products[p]['material'] for p in products)

print(f"\nResource usage:")
print(f"Labor used: {total_labor:.0f}/{available_labor} hours")
print(f"Material used: {total_material:.0f}/{available_material} units") 