"""
Convex Optimization Examples
============================

- Convex optimization with CVXPY
- Portfolio optimization
- Support vector machines
- Linear regression with constraints
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import pandas as pd

# 1. Basic Convex Optimization with CVXPY
print("=== Basic Convex Optimization with CVXPY ===")

# Example 1: Linear programming
print("Linear Programming Example:")
# Minimize: 2x + 3y
# Subject to: x + y >= 1, x >= 0, y >= 0

x = cp.Variable(2)
objective = cp.Minimize(2*x[0] + 3*x[1])
constraints = [x[0] + x[1] >= 1, x >= 0]
problem = cp.Problem(objective, constraints)

result = problem.solve()
print(f"Optimal value: {result:.4f}")
print(f"Optimal x: {x.value}")

# Example 2: Quadratic programming
print("\nQuadratic Programming Example:")
# Minimize: x^2 + y^2
# Subject to: x + y >= 1

x_qp = cp.Variable(2)
objective_qp = cp.Minimize(cp.sum_squares(x_qp))
constraints_qp = [cp.sum(x_qp) >= 1]
problem_qp = cp.Problem(objective_qp, constraints_qp)

result_qp = problem_qp.solve()
print(f"Optimal value: {result_qp:.4f}")
print(f"Optimal x: {x_qp.value}")

# Example 3: Second-order cone programming
print("\nSecond-Order Cone Programming Example:")
# Minimize: x + y
# Subject to: ||[x, y]||_2 <= 1

x_socp = cp.Variable(2)
objective_socp = cp.Minimize(cp.sum(x_socp))
constraints_socp = [cp.norm(x_socp, 2) <= 1]
problem_socp = cp.Problem(objective_socp, constraints_socp)

result_socp = problem_socp.solve()
print(f"Optimal value: {result_socp:.4f}")
print(f"Optimal x: {x_socp.value}")

# 2. Portfolio Optimization
print("\n=== Portfolio Optimization ===")

class PortfolioOptimizer:
    def __init__(self, returns, risk_free_rate=0.02):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.n_assets = len(returns.columns)
    
    def minimum_variance_portfolio(self):
        """Find minimum variance portfolio"""
        weights = cp.Variable(self.n_assets)
        
        # Objective: minimize portfolio variance
        portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
        objective = cp.Minimize(portfolio_variance)
        
        # Constraints: weights sum to 1, no short selling
        constraints = [cp.sum(weights) == 1, weights >= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return weights.value
    
    def maximum_sharpe_portfolio(self):
        """Find maximum Sharpe ratio portfolio"""
        weights = cp.Variable(self.n_assets)
        
        # Portfolio return and variance
        portfolio_return = self.mean_returns.values @ weights
        portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
        
        # Sharpe ratio (maximize return - risk_free_rate / std)
        # We minimize the negative Sharpe ratio
        objective = cp.Minimize(-(portfolio_return - self.risk_free_rate) / cp.sqrt(portfolio_variance))
        
        # Constraints: weights sum to 1, no short selling
        constraints = [cp.sum(weights) == 1, weights >= 0]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        return weights.value
    
    def efficient_frontier(self, target_returns):
        """Generate efficient frontier"""
        efficient_weights = []
        efficient_risks = []
        
        for target_return in target_returns:
            weights = cp.Variable(self.n_assets)
            
            # Objective: minimize portfolio variance
            portfolio_variance = cp.quad_form(weights, self.cov_matrix.values)
            objective = cp.Minimize(portfolio_variance)
            
            # Constraints: weights sum to 1, target return, no short selling
            portfolio_return = self.mean_returns.values @ weights
            constraints = [cp.sum(weights) == 1, 
                         portfolio_return >= target_return, 
                         weights >= 0]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                efficient_weights.append(weights.value)
                efficient_risks.append(np.sqrt(portfolio_variance.value))
        
        return efficient_weights, efficient_risks

# Generate sample stock returns
np.random.seed(42)
n_assets = 5
n_periods = 252  # One year of daily data

# Generate correlated returns
returns_data = np.random.multivariate_normal(
    mean=[0.08, 0.12, 0.15, 0.10, 0.09],  # Annual returns
    cov=[[0.04, 0.02, 0.01, 0.015, 0.02],
         [0.02, 0.09, 0.03, 0.025, 0.03],
         [0.01, 0.03, 0.16, 0.02, 0.025],
         [0.015, 0.025, 0.02, 0.06, 0.015],
         [0.02, 0.03, 0.025, 0.015, 0.05]],
    size=n_periods
)

# Convert to daily returns
daily_returns = returns_data / np.sqrt(252)
returns_df = pd.DataFrame(daily_returns, columns=['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E'])

# Test portfolio optimization
portfolio_opt = PortfolioOptimizer(returns_df)

# Minimum variance portfolio
min_var_weights = portfolio_opt.minimum_variance_portfolio()
min_var_return = returns_df.mean().values @ min_var_weights
min_var_risk = np.sqrt(min_var_weights @ portfolio_opt.cov_matrix.values @ min_var_weights)

print(f"Minimum Variance Portfolio:")
print(f"Weights: {min_var_weights}")
print(f"Expected Return: {min_var_return:.4f}")
print(f"Risk: {min_var_risk:.4f}")

# Maximum Sharpe ratio portfolio
max_sharpe_weights = portfolio_opt.maximum_sharpe_portfolio()
max_sharpe_return = returns_df.mean().values @ max_sharpe_weights
max_sharpe_risk = np.sqrt(max_sharpe_weights @ portfolio_opt.cov_matrix.values @ max_sharpe_weights)
max_sharpe_ratio = (max_sharpe_return - portfolio_opt.risk_free_rate) / max_sharpe_risk

print(f"\nMaximum Sharpe Ratio Portfolio:")
print(f"Weights: {max_sharpe_weights}")
print(f"Expected Return: {max_sharpe_return:.4f}")
print(f"Risk: {max_sharpe_risk:.4f}")
print(f"Sharpe Ratio: {max_sharpe_ratio:.4f}")

# Efficient frontier
target_returns = np.linspace(0.05, 0.15, 20)
efficient_weights, efficient_risks = portfolio_opt.efficient_frontier(target_returns)

# 3. Support Vector Machines with Convex Optimization
print("\n=== Support Vector Machines with Convex Optimization ===")

class SVMClassifier:
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None
        self.support_vectors = None
    
    def fit(self, X, y):
        """Fit SVM using convex optimization"""
        n_samples, n_features = X.shape
        
        # Variables
        w = cp.Variable(n_features)
        b = cp.Variable()
        xi = cp.Variable(n_samples)  # Slack variables
        
        # Objective: minimize ||w||^2 + C * sum(xi)
        objective = cp.Minimize(0.5 * cp.sum_squares(w) + self.C * cp.sum(xi))
        
        # Constraints: y_i * (w^T * x_i + b) >= 1 - xi_i, xi_i >= 0
        constraints = []
        for i in range(n_samples):
            constraints.append(y[i] * (X[i] @ w + b) >= 1 - xi[i])
        constraints.append(xi >= 0)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        self.w = w.value
        self.b = b.value
        
        # Find support vectors
        margins = y * (X @ self.w + self.b)
        self.support_vectors = X[np.abs(margins - 1) < 1e-5]
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        return np.sign(X @ self.w + self.b)

# Test SVM
X_svm, y_svm = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                  n_informative=2, random_state=42, n_clusters_per_class=1)
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.3, random_state=42)

svm_opt = SVMClassifier(C=1.0)
svm_opt.fit(X_train_svm, y_train_svm)
y_pred_svm = svm_opt.predict(X_test_svm)
svm_accuracy = accuracy_score(y_test_svm, y_pred_svm)

print(f"SVM Accuracy: {svm_accuracy:.4f}")
print(f"Number of support vectors: {len(svm_opt.support_vectors)}")

# 4. Linear Regression with Constraints
print("\n=== Linear Regression with Constraints ===")

class ConstrainedLinearRegression:
    def __init__(self, constraints=None):
        self.constraints = constraints or []
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y, bounds=None):
        """Fit linear regression with constraints"""
        n_samples, n_features = X.shape
        
        # Variables
        beta = cp.Variable(n_features)
        intercept = cp.Variable()
        
        # Objective: minimize ||y - X*beta - intercept||^2
        objective = cp.Minimize(cp.sum_squares(y - X @ beta - intercept))
        
        # Constraints
        constraints = []
        
        # Add user-defined constraints
        for constraint in self.constraints:
            constraints.append(constraint(beta, intercept))
        
        # Add bounds if specified
        if bounds is not None:
            for i, (lower, upper) in enumerate(bounds):
                if lower is not None:
                    constraints.append(beta[i] >= lower)
                if upper is not None:
                    constraints.append(beta[i] <= upper)
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        self.coefficients = beta.value
        self.intercept = intercept.value
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.coefficients + self.intercept

# Test constrained linear regression
X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Define constraints
def non_negative_constraint(beta, intercept):
    """All coefficients must be non-negative"""
    return beta >= 0

def sum_constraint(beta, intercept):
    """Sum of coefficients must equal 1"""
    return cp.sum(beta) == 1

# Test different constraint combinations
constraint_configs = [
    ("No constraints", []),
    ("Non-negative coefficients", [non_negative_constraint]),
    ("Sum equals 1", [sum_constraint]),
    ("Both constraints", [non_negative_constraint, sum_constraint])
]

regression_results = {}
for name, constraints in constraint_configs:
    print(f"\n--- {name} ---")
    
    clr = ConstrainedLinearRegression(constraints=constraints)
    clr.fit(X_train_reg, y_train_reg)
    
    y_pred_reg = clr.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    
    regression_results[name] = {
        'coefficients': clr.coefficients,
        'intercept': clr.intercept,
        'mse': mse
    }
    
    print(f"Coefficients: {clr.coefficients}")
    print(f"Intercept: {clr.intercept:.4f}")
    print(f"MSE: {mse:.4f}")

# 5. Advanced Convex Optimization Examples
print("\n=== Advanced Convex Optimization Examples ===")

# Example 1: Lasso regression
print("Lasso Regression:")
X_lasso, y_lasso = make_regression(n_samples=50, n_features=10, noise=0.1, random_state=42)

beta_lasso = cp.Variable(10)
lambda_param = 0.1

# Objective: minimize ||y - X*beta||^2 + lambda * ||beta||_1
objective_lasso = cp.Minimize(cp.sum_squares(y_lasso - X_lasso @ beta_lasso) + 
                             lambda_param * cp.norm(beta_lasso, 1))

problem_lasso = cp.Problem(objective_lasso)
problem_lasso.solve()

print(f"Lasso coefficients: {beta_lasso.value}")

# Example 2: Ridge regression
print("\nRidge Regression:")
beta_ridge = cp.Variable(10)
lambda_ridge = 0.1

# Objective: minimize ||y - X*beta||^2 + lambda * ||beta||^2
objective_ridge = cp.Minimize(cp.sum_squares(y_lasso - X_lasso @ beta_ridge) + 
                             lambda_ridge * cp.sum_squares(beta_ridge))

problem_ridge = cp.Problem(objective_ridge)
problem_ridge.solve()

print(f"Ridge coefficients: {beta_ridge.value}")

# Example 3: Elastic net
print("\nElastic Net:")
beta_elastic = cp.Variable(10)
lambda1 = 0.1  # L1 penalty
lambda2 = 0.1  # L2 penalty

# Objective: minimize ||y - X*beta||^2 + lambda1 * ||beta||_1 + lambda2 * ||beta||^2
objective_elastic = cp.Minimize(cp.sum_squares(y_lasso - X_lasso @ beta_elastic) + 
                               lambda1 * cp.norm(beta_elastic, 1) + 
                               lambda2 * cp.sum_squares(beta_elastic))

problem_elastic = cp.Problem(objective_elastic)
problem_elastic.solve()

print(f"Elastic net coefficients: {beta_elastic.value}")

# 6. Visualization
print("\n=== Visualization ===")

plt.figure(figsize=(15, 10))

# Plot efficient frontier
plt.subplot(2, 3, 1)
if efficient_risks:
    plt.plot(efficient_risks, target_returns[:len(efficient_risks)], 'b-', linewidth=2, label='Efficient Frontier')
    plt.scatter(min_var_risk, min_var_return, color='red', s=100, label='Min Variance')
    plt.scatter(max_sharpe_risk, max_sharpe_return, color='green', s=100, label='Max Sharpe')
    plt.xlabel('Portfolio Risk')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    plt.grid(True)

# Plot SVM decision boundary
plt.subplot(2, 3, 2)
plt.scatter(X_train_svm[:, 0], X_train_svm[:, 1], c=y_train_svm, cmap='viridis', alpha=0.7)
if svm_opt.support_vectors is not None:
    plt.scatter(svm_opt.support_vectors[:, 0], svm_opt.support_vectors[:, 1], 
               s=100, facecolors='none', edgecolors='red', label='Support Vectors')

# Decision boundary
x_min, x_max = X_train_svm[:, 0].min() - 1, X_train_svm[:, 0].max() + 1
y_min, y_max = X_train_svm[:, 1].min() - 1, X_train_svm[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = svm_opt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.legend()

# Plot regression coefficients comparison
plt.subplot(2, 3, 3)
reg_names = list(regression_results.keys())
reg_coeffs = [regression_results[name]['coefficients'] for name in reg_names]
reg_coeffs_array = np.array(reg_coeffs)

x_pos = np.arange(len(reg_names))
width = 0.15

for i in range(reg_coeffs_array.shape[1]):
    plt.bar(x_pos + i*width, reg_coeffs_array[:, i], width, 
            label=f'Coeff {i+1}', alpha=0.8)

plt.xlabel('Constraint Configuration')
plt.ylabel('Coefficient Value')
plt.title('Regression Coefficients Comparison')
plt.xticks(x_pos + width*2, reg_names, rotation=45)
plt.legend()
plt.grid(True)

# Plot regularization comparison
plt.subplot(2, 3, 4)
coeff_names = ['Lasso', 'Ridge', 'Elastic Net']
coeff_values = [beta_lasso.value, beta_ridge.value, beta_elastic.value]

x_pos = np.arange(len(coeff_names))
width = 0.15

for i in range(10):
    plt.bar(x_pos + i*width, [coeff_values[0][i], coeff_values[1][i], coeff_values[2][i]], 
            width, label=f'Feature {i+1}', alpha=0.8)

plt.xlabel('Regularization Method')
plt.ylabel('Coefficient Value')
plt.title('Regularization Comparison')
plt.xticks(x_pos + width*4.5, coeff_names)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Plot portfolio weights
plt.subplot(2, 3, 5)
asset_names = ['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 'Stock_E']
x_pos = np.arange(len(asset_names))
width = 0.35

plt.bar(x_pos - width/2, min_var_weights, width, label='Min Variance', alpha=0.8)
plt.bar(x_pos + width/2, max_sharpe_weights, width, label='Max Sharpe', alpha=0.8)

plt.xlabel('Assets')
plt.ylabel('Weight')
plt.title('Portfolio Weights Comparison')
plt.xticks(x_pos, asset_names, rotation=45)
plt.legend()
plt.grid(True)

# Plot MSE comparison
plt.subplot(2, 3, 6)
mse_values = [regression_results[name]['mse'] for name in reg_names]
plt.bar(reg_names, mse_values, alpha=0.8)
plt.xlabel('Constraint Configuration')
plt.ylabel('Mean Squared Error')
plt.title('Regression MSE Comparison')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. Performance summary
print("\n=== Performance Summary ===")
print("Portfolio Optimization:")
print(f"Minimum Variance Portfolio - Return: {min_var_return:.4f}, Risk: {min_var_risk:.4f}")
print(f"Maximum Sharpe Portfolio - Return: {max_sharpe_return:.4f}, Risk: {max_sharpe_risk:.4f}, Sharpe: {max_sharpe_ratio:.4f}")

print(f"\nSVM Classification:")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Support Vectors: {len(svm_opt.support_vectors)}")

print(f"\nConstrained Linear Regression MSE:")
for name, result in regression_results.items():
    print(f"{name}: {result['mse']:.4f}")

print(f"\nRegularization Comparison:")
print(f"Lasso - Non-zero coefficients: {np.sum(np.abs(beta_lasso.value) > 1e-6)}")
print(f"Ridge - Non-zero coefficients: {np.sum(np.abs(beta_ridge.value) > 1e-6)}")
print(f"Elastic Net - Non-zero coefficients: {np.sum(np.abs(beta_elastic.value) > 1e-6)}")

print("\n=== Summary ===")
print("1. Basic Convex Optimization: Linear, quadratic, and SOCP examples")
print("2. Portfolio Optimization: Minimum variance and maximum Sharpe ratio portfolios")
print("3. SVM Classification: Linear SVM with convex optimization")
print("4. Constrained Linear Regression: Various constraint configurations")
print("5. Advanced Examples: Lasso, Ridge, and Elastic Net regularization")
print("6. Comprehensive visualization of all results") 