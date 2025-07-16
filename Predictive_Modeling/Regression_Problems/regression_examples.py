"""
Regression Problems Examples
============================

- House Price Prediction
- Salary Prediction
- Product Rating Prediction
- Model comparison and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. House Price Prediction
print("=== House Price Prediction ===")

def generate_house_data(n_houses=2000):
    """Generate synthetic house price data"""
    np.random.seed(42)
    
    # Generate house features
    square_feet = np.random.normal(2000, 500, n_houses)
    square_feet = np.clip(square_feet, 800, 4000)
    
    bedrooms = np.random.poisson(3, n_houses)
    bedrooms = np.clip(bedrooms, 1, 6)
    
    bathrooms = np.random.poisson(2, n_houses)
    bathrooms = np.clip(bathrooms, 1, 4)
    
    year_built = np.random.randint(1960, 2023, n_houses)
    
    # Generate categorical features
    location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_houses, p=[0.4, 0.4, 0.2])
    condition = np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_houses, p=[0.2, 0.4, 0.3, 0.1])
    
    # Generate price based on features
    base_price = 200000
    
    # Square feet effect
    price_per_sqft = 150 + np.random.normal(0, 20, n_houses)
    price = square_feet * price_per_sqft
    
    # Bedrooms effect
    price += bedrooms * 25000
    
    # Bathrooms effect
    price += bathrooms * 30000
    
    # Year built effect (newer houses cost more)
    price += (year_built - 1960) * 1000
    
    # Location effect
    location_multiplier = np.where(location == 'Urban', 1.3, 
                                  np.where(location == 'Suburban', 1.1, 0.8))
    price *= location_multiplier
    
    # Condition effect
    condition_multiplier = np.where(condition == 'Excellent', 1.2,
                                   np.where(condition == 'Good', 1.0,
                                           np.where(condition == 'Fair', 0.8, 0.6)))
    price *= condition_multiplier
    
    # Add some randomness
    price += np.random.normal(0, price * 0.1, n_houses)
    price = np.maximum(price, 50000)  # Minimum price
    
    return pd.DataFrame({
        'house_id': range(1, n_houses + 1),
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'year_built': year_built,
        'location': location,
        'condition': condition,
        'price': price
    })

# Generate house data
house_data = generate_house_data()
print(f"House data shape: {house_data.shape}")
print(f"Price range: ${house_data['price'].min():,.0f} - ${house_data['price'].max():,.0f}")
print(f"Average price: ${house_data['price'].mean():,.0f}")

# Prepare features
house_features = ['square_feet', 'bedrooms', 'bathrooms', 'year_built', 'location', 'condition']
X_house = house_data[house_features].copy()
y_house = house_data['price']

# Encode categorical variables
le_location = LabelEncoder()
le_condition = LabelEncoder()

X_house['location'] = le_location.fit_transform(X_house['location'])
X_house['condition'] = le_condition.fit_transform(X_house['condition'])

# Split data
X_train_house, X_test_house, y_train_house, y_test_house = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Scale features
scaler_house = StandardScaler()
X_train_scaled_house = scaler_house.fit_transform(X_train_house)
X_test_scaled_house = scaler_house.transform(X_test_house)

# Train multiple models
models_house = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'SVR': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
}

house_results = {}
for name, model in models_house.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled_house, y_train_house)
    y_pred = model.predict(X_test_scaled_house)
    
    house_results[name] = {
        'rmse': np.sqrt(mean_squared_error(y_test_house, y_pred)),
        'mae': mean_absolute_error(y_test_house, y_pred),
        'r2': r2_score(y_test_house, y_pred),
        'predictions': y_pred
    }

# 2. Salary Prediction
print("\n=== Salary Prediction ===")

def generate_salary_data(n_employees=3000):
    """Generate synthetic salary data"""
    np.random.seed(42)
    
    # Generate employee features
    age = np.random.normal(35, 10, n_employees)
    age = np.clip(age, 22, 65)
    
    experience_years = np.random.exponential(scale=8, size=n_employees)
    experience_years = np.clip(experience_years, 0, 30)
    
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                     n_employees, p=[0.2, 0.5, 0.2, 0.1])
    
    job_title = np.random.choice(['Software Engineer', 'Data Scientist', 'Manager', 'Analyst', 'Director'], 
                               n_employees, p=[0.3, 0.2, 0.2, 0.2, 0.1])
    
    company_size = np.random.choice(['Small', 'Medium', 'Large'], n_employees, p=[0.3, 0.4, 0.3])
    
    # Generate salary based on features
    base_salary = 50000
    
    # Age effect (diminishing returns)
    age_effect = np.minimum(age - 22, 20) * 1000
    
    # Experience effect
    experience_effect = experience_years * 3000
    
    # Education effect
    education_multiplier = np.where(education_level == 'High School', 1.0,
                                   np.where(education_level == 'Bachelor', 1.3,
                                           np.where(education_level == 'Master', 1.6, 2.0)))
    
    # Job title effect
    title_multiplier = np.where(job_title == 'Software Engineer', 1.2,
                               np.where(job_title == 'Data Scientist', 1.4,
                                       np.where(job_title == 'Manager', 1.6,
                                               np.where(job_title == 'Analyst', 1.1, 2.0))))
    
    # Company size effect
    size_multiplier = np.where(company_size == 'Small', 0.8,
                              np.where(company_size == 'Medium', 1.0, 1.3))
    
    # Calculate salary
    salary = (base_salary + age_effect + experience_effect) * education_multiplier * title_multiplier * size_multiplier
    
    # Add some randomness
    salary += np.random.normal(0, salary * 0.15, n_employees)
    salary = np.maximum(salary, 30000)  # Minimum salary
    
    return pd.DataFrame({
        'employee_id': range(1, n_employees + 1),
        'age': age,
        'experience_years': experience_years,
        'education_level': education_level,
        'job_title': job_title,
        'company_size': company_size,
        'salary': salary
    })

# Generate salary data
salary_data = generate_salary_data()
print(f"Salary data shape: {salary_data.shape}")
print(f"Salary range: ${salary_data['salary'].min():,.0f} - ${salary_data['salary'].max():,.0f}")
print(f"Average salary: ${salary_data['salary'].mean():,.0f}")

# Prepare features
salary_features = ['age', 'experience_years', 'education_level', 'job_title', 'company_size']
X_salary = salary_data[salary_features].copy()
y_salary = salary_data['salary']

# Encode categorical variables
le_education = LabelEncoder()
le_job_title = LabelEncoder()
le_company_size = LabelEncoder()

X_salary['education_level'] = le_education.fit_transform(X_salary['education_level'])
X_salary['job_title'] = le_job_title.fit_transform(X_salary['job_title'])
X_salary['company_size'] = le_company_size.fit_transform(X_salary['company_size'])

# Split data
X_train_salary, X_test_salary, y_train_salary, y_test_salary = train_test_split(
    X_salary, y_salary, test_size=0.2, random_state=42
)

# Scale features
scaler_salary = StandardScaler()
X_train_scaled_salary = scaler_salary.fit_transform(X_train_salary)
X_test_scaled_salary = scaler_salary.transform(X_test_salary)

# Train models for salary prediction
salary_results = {}
for name, model in models_house.items():
    print(f"\nTraining {name} for salary prediction...")
    model.fit(X_train_scaled_salary, y_train_salary)
    y_pred = model.predict(X_test_scaled_salary)
    
    salary_results[name] = {
        'rmse': np.sqrt(mean_squared_error(y_test_salary, y_pred)),
        'mae': mean_absolute_error(y_test_salary, y_pred),
        'r2': r2_score(y_test_salary, y_pred),
        'predictions': y_pred
    }

# 3. Product Rating Prediction
print("\n=== Product Rating Prediction ===")

def generate_product_data(n_products=2500):
    """Generate synthetic product rating data"""
    np.random.seed(42)
    
    # Generate product features
    price = np.random.exponential(scale=50, size=n_products)
    price = np.clip(price, 10, 500)
    
    weight = np.random.normal(1.5, 0.5, n_products)
    weight = np.clip(weight, 0.1, 5.0)
    
    shipping_time = np.random.exponential(scale=3, size=n_products)
    shipping_time = np.clip(shipping_time, 1, 14)
    
    # Generate categorical features
    category = np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], 
                              n_products, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    brand = np.random.choice(['Premium', 'Standard', 'Budget'], n_products, p=[0.2, 0.5, 0.3])
    
    availability = np.random.choice(['In Stock', 'Limited', 'Out of Stock'], n_products, p=[0.7, 0.2, 0.1])
    
    # Generate rating based on features
    base_rating = 3.0
    
    # Price effect (optimal price range)
    price_effect = np.where(price < 30, -0.5,
                           np.where(price > 200, -0.3, 0.2))
    
    # Weight effect (lighter is better for shipping)
    weight_effect = np.where(weight > 3, -0.3, 0.1)
    
    # Shipping time effect
    shipping_effect = np.where(shipping_time > 7, -0.4,
                              np.where(shipping_time > 3, -0.1, 0.2))
    
    # Category effect
    category_effect = np.where(category == 'Electronics', 0.2,
                              np.where(category == 'Books', 0.1,
                                      np.where(category == 'Sports', 0.3, 0.0)))
    
    # Brand effect
    brand_effect = np.where(brand == 'Premium', 0.4,
                           np.where(brand == 'Standard', 0.1, -0.2))
    
    # Availability effect
    availability_effect = np.where(availability == 'In Stock', 0.2,
                                  np.where(availability == 'Limited', 0.0, -0.5))
    
    # Calculate rating
    rating = base_rating + price_effect + weight_effect + shipping_effect + category_effect + brand_effect + availability_effect
    
    # Add some randomness
    rating += np.random.normal(0, 0.3, n_products)
    rating = np.clip(rating, 1.0, 5.0)  # Rating between 1-5
    
    return pd.DataFrame({
        'product_id': range(1, n_products + 1),
        'price': price,
        'weight': weight,
        'shipping_time': shipping_time,
        'category': category,
        'brand': brand,
        'availability': availability,
        'rating': rating
    })

# Generate product data
product_data = generate_product_data()
print(f"Product data shape: {product_data.shape}")
print(f"Rating range: {product_data['rating'].min():.1f} - {product_data['rating'].max():.1f}")
print(f"Average rating: {product_data['rating'].mean():.2f}")

# Prepare features
product_features = ['price', 'weight', 'shipping_time', 'category', 'brand', 'availability']
X_product = product_data[product_features].copy()
y_product = product_data['rating']

# Encode categorical variables
le_category = LabelEncoder()
le_brand = LabelEncoder()
le_availability = LabelEncoder()

X_product['category'] = le_category.fit_transform(X_product['category'])
X_product['brand'] = le_brand.fit_transform(X_product['brand'])
X_product['availability'] = le_availability.fit_transform(X_product['availability'])

# Split data
X_train_product, X_test_product, y_train_product, y_test_product = train_test_split(
    X_product, y_product, test_size=0.2, random_state=42
)

# Scale features
scaler_product = StandardScaler()
X_train_scaled_product = scaler_product.fit_transform(X_train_product)
X_test_scaled_product = scaler_product.transform(X_test_product)

# Train models for product rating prediction
product_results = {}
for name, model in models_house.items():
    print(f"\nTraining {name} for product rating prediction...")
    model.fit(X_train_scaled_product, y_train_product)
    y_pred = model.predict(X_test_scaled_product)
    
    product_results[name] = {
        'rmse': np.sqrt(mean_squared_error(y_test_product, y_pred)),
        'mae': mean_absolute_error(y_test_product, y_pred),
        'r2': r2_score(y_test_product, y_pred),
        'predictions': y_pred
    }

# 4. Model Comparison and Visualization
print("\n=== Model Comparison and Visualization ===")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Function to plot results
def plot_regression_results(results, title, ax, metric='r2'):
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    bars = ax.bar(models, values, alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{title} - {metric.upper()}')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

# Plot R² scores
plot_regression_results(house_results, 'House Price Prediction', axes[0, 0], 'r2')
plot_regression_results(salary_results, 'Salary Prediction', axes[0, 1], 'r2')
plot_regression_results(product_results, 'Product Rating Prediction', axes[0, 2], 'r2')

# Plot RMSE scores
plot_regression_results(house_results, 'House Price Prediction', axes[1, 0], 'rmse')
plot_regression_results(salary_results, 'Salary Prediction', axes[1, 1], 'rmse')
plot_regression_results(product_results, 'Product Rating Prediction', axes[1, 2], 'rmse')

plt.tight_layout()
plt.show()

# 5. Prediction vs Actual Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Function to plot predictions vs actual
def plot_predictions_vs_actual(results, y_true, title, ax, best_model_name):
    y_pred = results[best_model_name]['predictions']
    
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{title} - {best_model_name}')
    ax.grid(True, alpha=0.3)

# Find best models
def find_best_model(results, metric='r2'):
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0]

best_house_model = find_best_model(house_results)
best_salary_model = find_best_model(salary_results)
best_product_model = find_best_model(product_results)

# Plot predictions vs actual
plot_predictions_vs_actual(house_results, y_test_house, 'House Price Prediction', axes[0, 0], best_house_model)
plot_predictions_vs_actual(salary_results, y_test_salary, 'Salary Prediction', axes[0, 1], best_salary_model)
plot_predictions_vs_actual(product_results, y_test_product, 'Product Rating Prediction', axes[0, 2], best_product_model)

# 6. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# Get feature importance from Random Forest models
rf_house = models_house['Random Forest']
rf_salary = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled_salary, y_train_salary)
rf_product = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_scaled_product, y_train_product)

# Plot feature importance
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# House feature importance
house_importance = pd.DataFrame({
    'feature': house_features,
    'importance': rf_house.feature_importances_
}).sort_values('importance', ascending=True)

axes[0].barh(house_importance['feature'], house_importance['importance'])
axes[0].set_title('Feature Importance - House Price Prediction')
axes[0].set_xlabel('Importance')

# Salary feature importance
salary_importance = pd.DataFrame({
    'feature': salary_features,
    'importance': rf_salary.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(salary_importance['feature'], salary_importance['importance'])
axes[1].set_title('Feature Importance - Salary Prediction')
axes[1].set_xlabel('Importance')

# Product feature importance
product_importance = pd.DataFrame({
    'feature': product_features,
    'importance': rf_product.feature_importances_
}).sort_values('importance', ascending=True)

axes[2].barh(product_importance['feature'], product_importance['importance'])
axes[2].set_title('Feature Importance - Product Rating Prediction')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# 7. Residual Analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Function to plot residuals
def plot_residuals(results, y_true, title, ax, best_model_name):
    y_pred = results[best_model_name]['predictions']
    residuals = y_true - y_pred
    
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(f'{title} - Residuals ({best_model_name})')
    ax.grid(True, alpha=0.3)

# Plot residuals
plot_residuals(house_results, y_test_house, 'House Price Prediction', axes[0, 0], best_house_model)
plot_residuals(salary_results, y_test_salary, 'Salary Prediction', axes[0, 1], best_salary_model)
plot_residuals(product_results, y_test_product, 'Product Rating Prediction', axes[0, 2], best_product_model)

# 8. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

# Find best model for each problem
def find_best_model_detailed(results, metric='r2'):
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1]

print("Best Models by R² Score:")
print(f"1. House Price Prediction:")
best_house, house_metrics = find_best_model_detailed(house_results)
print(f"   Best model: {best_house}")
print(f"   R²: {house_metrics['r2']:.3f}")
print(f"   RMSE: ${house_metrics['rmse']:,.0f}")
print(f"   MAE: ${house_metrics['mae']:,.0f}")

print(f"\n2. Salary Prediction:")
best_salary, salary_metrics = find_best_model_detailed(salary_results)
print(f"   Best model: {best_salary}")
print(f"   R²: {salary_metrics['r2']:.3f}")
print(f"   RMSE: ${salary_metrics['rmse']:,.0f}")
print(f"   MAE: ${salary_metrics['mae']:,.0f}")

print(f"\n3. Product Rating Prediction:")
best_product, product_metrics = find_best_model_detailed(product_results)
print(f"   Best model: {best_product}")
print(f"   R²: {product_metrics['r2']:.3f}")
print(f"   RMSE: {product_metrics['rmse']:.3f}")
print(f"   MAE: {product_metrics['mae']:.3f}")

print(f"\nKey Insights:")
print(f"- Random Forest and Gradient Boosting perform well across all regression problems")
print(f"- Linear models (Ridge, Lasso) provide good baseline performance")
print(f"- Neural Networks show competitive performance for complex patterns")
print(f"- Feature engineering and scaling are crucial for good performance")

print(f"\nRecommendations:")
print(f"- Use Random Forest for balanced performance and interpretability")
print(f"- Use Gradient Boosting for maximum predictive performance")
print(f"- Use Ridge/Lasso for interpretable models with regularization")
print(f"- Always check for multicollinearity in linear models")
print(f"- Consider feature interactions for complex relationships")
print(f"- Validate model assumptions (normality, homoscedasticity) for linear models") 