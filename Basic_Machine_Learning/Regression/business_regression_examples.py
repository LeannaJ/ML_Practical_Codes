"""
Business Regression Examples
===========================

This script demonstrates regression analysis for real-world business scenarios including:
- Sales Prediction
- Customer Lifetime Value Prediction
- Price Optimization
- Demand Forecasting
- Employee Performance Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_sales_prediction_data():
    """Create sample data for sales prediction"""
    n_samples = 1000
    
    # Generate features
    advertising_budget = np.random.exponential(5000, n_samples)
    social_media_followers = np.random.poisson(10000, n_samples)
    competitor_price = np.random.normal(50, 10, n_samples)
    season = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples)
    day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples)
    
    # Create sales target with realistic relationships
    base_sales = 1000
    sales = (base_sales + 
             0.1 * advertising_budget + 
             0.05 * social_media_followers - 
             2 * competitor_price +
             np.random.normal(0, 100, n_samples))
    
    # Add seasonal effects
    seasonal_effects = {'Spring': 1.1, 'Summer': 1.2, 'Fall': 0.9, 'Winter': 0.8}
    for i, s in enumerate(season):
        sales[i] *= seasonal_effects[s]
    
    # Add weekend effects
    weekend_effects = {'Monday': 0.9, 'Tuesday': 0.95, 'Wednesday': 1.0, 
                      'Thursday': 1.05, 'Friday': 1.1, 'Saturday': 1.3, 'Sunday': 1.2}
    for i, d in enumerate(day_of_week):
        sales[i] *= weekend_effects[d]
    
    # Ensure positive sales
    sales = np.maximum(sales, 0)
    
    data = {
        'advertising_budget': advertising_budget,
        'social_media_followers': social_media_followers,
        'competitor_price': competitor_price,
        'season': season,
        'day_of_week': day_of_week,
        'sales': sales
    }
    
    return pd.DataFrame(data)

def create_customer_lifetime_value_data():
    """Create sample data for customer lifetime value prediction"""
    n_samples = 1000
    
    # Generate customer features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.exponential(50000, n_samples)
    education_years = np.random.choice([12, 16, 18, 22], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    credit_score = np.random.normal(700, 100, n_samples)
    years_customer = np.random.exponential(3, n_samples)
    avg_purchase_value = np.random.normal(100, 30, n_samples)
    purchase_frequency = np.random.poisson(5, n_samples)
    customer_service_calls = np.random.poisson(2, n_samples)
    
    # Create CLV target
    clv = (income * 0.01 + 
           avg_purchase_value * purchase_frequency * 0.5 +
           years_customer * 1000 -
           customer_service_calls * 500 +
           np.random.normal(0, 2000, n_samples))
    
    # Ensure positive CLV
    clv = np.maximum(clv, 0)
    
    data = {
        'age': age,
        'income': income,
        'education_years': education_years,
        'credit_score': credit_score,
        'years_customer': years_customer,
        'avg_purchase_value': avg_purchase_value,
        'purchase_frequency': purchase_frequency,
        'customer_service_calls': customer_service_calls,
        'customer_lifetime_value': clv
    }
    
    return pd.DataFrame(data)

def create_price_optimization_data():
    """Create sample data for price optimization"""
    n_samples = 1000
    
    # Generate features
    product_cost = np.random.normal(20, 5, n_samples)
    competitor_price = np.random.normal(50, 10, n_samples)
    market_demand = np.random.poisson(1000, n_samples)
    season = np.random.choice(['Low', 'Medium', 'High'], n_samples)
    product_rating = np.random.uniform(3.0, 5.0, n_samples)
    
    # Create optimal price target (simplified pricing model)
    optimal_price = (product_cost * 2.5 + 
                    competitor_price * 0.3 +
                    market_demand * 0.01 +
                    product_rating * 5 +
                    np.random.normal(0, 5, n_samples))
    
    # Ensure reasonable price range
    optimal_price = np.clip(optimal_price, product_cost * 1.2, product_cost * 5)
    
    data = {
        'product_cost': product_cost,
        'competitor_price': competitor_price,
        'market_demand': market_demand,
        'season': season,
        'product_rating': product_rating,
        'optimal_price': optimal_price
    }
    
    return pd.DataFrame(data)

def create_demand_forecasting_data():
    """Create sample data for demand forecasting"""
    n_samples = 1000
    
    # Generate time series features
    month = np.random.randint(1, 13, n_samples)
    day_of_month = np.random.randint(1, 29, n_samples)
    day_of_week = np.random.randint(0, 7, n_samples)
    is_holiday = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    temperature = np.random.normal(20, 10, n_samples)
    marketing_campaign = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Create demand target with seasonal and trend patterns
    base_demand = 1000
    
    # Seasonal effect (higher in summer, lower in winter)
    seasonal_effect = np.sin(2 * np.pi * month / 12) * 200
    
    # Weekly pattern (higher on weekends)
    weekly_effect = np.where(day_of_week >= 5, 100, 0)
    
    # Temperature effect
    temperature_effect = (temperature - 20) * 10
    
    # Holiday effect
    holiday_effect = is_holiday * 300
    
    # Marketing effect
    marketing_effect = marketing_campaign * 200
    
    demand = (base_demand + 
             seasonal_effect + 
             weekly_effect + 
             temperature_effect + 
             holiday_effect + 
             marketing_effect +
             np.random.normal(0, 50, n_samples))
    
    # Ensure positive demand
    demand = np.maximum(demand, 0)
    
    data = {
        'month': month,
        'day_of_month': day_of_month,
        'day_of_week': day_of_week,
        'is_holiday': is_holiday,
        'temperature': temperature,
        'marketing_campaign': marketing_campaign,
        'demand': demand
    }
    
    return pd.DataFrame(data)

def create_employee_performance_data():
    """Create sample data for employee performance prediction"""
    n_samples = 1000
    
    # Generate employee features
    age = np.random.normal(35, 8, n_samples)
    years_experience = np.random.exponential(5, n_samples)
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    training_hours = np.random.poisson(40, n_samples)
    projects_completed = np.random.poisson(10, n_samples)
    team_size = np.random.poisson(5, n_samples)
    salary = np.random.normal(60000, 20000, n_samples)
    
    # Create performance score target
    performance_score = (age * 0.5 + 
                        years_experience * 10 +
                        training_hours * 2 +
                        projects_completed * 5 +
                        np.random.normal(0, 10, n_samples))
    
    # Normalize to 0-100 scale
    performance_score = np.clip(performance_score, 0, 100)
    
    data = {
        'age': age,
        'years_experience': years_experience,
        'education_level': education_level,
        'training_hours': training_hours,
        'projects_completed': projects_completed,
        'team_size': team_size,
        'salary': salary,
        'performance_score': performance_score
    }
    
    return pd.DataFrame(data)

def prepare_data_for_modeling(df, target_column, categorical_columns=None):
    """Prepare data for modeling"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    if categorical_columns:
        X_encoded = X.copy()
        label_encoders = {}
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
    else:
        X_encoded = X
        label_encoders = {}
    
    # Remove any remaining non-numeric columns
    X_encoded = X_encoded.select_dtypes(include=[np.number])
    
    return X_encoded, y, label_encoders

def train_and_evaluate_models(X, y, problem_name):
    """Train and evaluate multiple models"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {problem_name.upper()}")
    print(f"{'='*60}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for linear models
        if 'Linear' in name or 'Ridge' in name or 'Lasso' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'CV_R2_mean': cv_mean,
            'CV_R2_std': cv_std,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  CV R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    return results, y_test

def plot_results(results, y_test, problem_name):
    """Plot model comparison and predictions"""
    print(f"\nPlotting results for {problem_name}...")
    
    # Model comparison
    models = list(results.keys())
    r2_scores = [results[model]['R2'] for model in models]
    rmse_scores = [results[model]['RMSE'] for model in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # R² Score comparison
    bars1 = axes[0].bar(models, r2_scores)
    axes[0].set_title(f'R² Score Comparison - {problem_name}')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    
    # RMSE comparison
    bars2 = axes[1].bar(models, rmse_scores)
    axes[1].set_title(f'RMSE Comparison - {problem_name}')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    
    # Predictions vs Actual (best model)
    best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    best_predictions = results[best_model_name]['predictions']
    
    axes[2].scatter(y_test, best_predictions, alpha=0.6)
    axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[2].set_xlabel('Actual Values')
    axes[2].set_ylabel('Predicted Values')
    axes[2].set_title(f'Best Model: {best_model_name}\nR² = {results[best_model_name]["R2"]:.3f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print ranking
    print(f"\nModel Ranking for {problem_name} (by R² Score):")
    ranking = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
    for i, (model, metrics) in enumerate(ranking, 1):
        print(f"{i}. {model}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")

def business_insights_analysis(df, target_column, problem_name):
    """Generate business insights from the data"""
    print(f"\n{'='*60}")
    print(f"BUSINESS INSIGHTS - {problem_name.upper()}")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"Target variable: {target_column}")
    print(f"Mean {target_column}: {df[target_column].mean():.2f}")
    print(f"Median {target_column}: {df[target_column].median():.2f}")
    print(f"Std {target_column}: {df[target_column].std():.2f}")
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_column].sort_values(ascending=False)
    
    print(f"\nTop correlations with {target_column}:")
    for feature, corr in correlations.head(6).items():
        if feature != target_column:
            print(f"  {feature}: {corr:.3f}")
    
    # Feature distributions
    print(f"\nFeature distributions:")
    for col in numeric_cols:
        if col != target_column:
            print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

def main():
    """Main function to run all business regression examples"""
    print("=== BUSINESS REGRESSION EXAMPLES ===\n")
    
    # 1. Sales Prediction
    print("1. Creating sales prediction dataset...")
    sales_df = create_sales_prediction_data()
    X_sales, y_sales, _ = prepare_data_for_modeling(sales_df, 'sales', ['season', 'day_of_week'])
    sales_results, y_test_sales = train_and_evaluate_models(X_sales, y_sales, "Sales Prediction")
    plot_results(sales_results, y_test_sales, "Sales Prediction")
    business_insights_analysis(sales_df, 'sales', "Sales Prediction")
    
    # 2. Customer Lifetime Value Prediction
    print("\n2. Creating customer lifetime value dataset...")
    clv_df = create_customer_lifetime_value_data()
    X_clv, y_clv, _ = prepare_data_for_modeling(clv_df, 'customer_lifetime_value')
    clv_results, y_test_clv = train_and_evaluate_models(X_clv, y_clv, "Customer Lifetime Value")
    plot_results(clv_results, y_test_clv, "Customer Lifetime Value")
    business_insights_analysis(clv_df, 'customer_lifetime_value', "Customer Lifetime Value")
    
    # 3. Price Optimization
    print("\n3. Creating price optimization dataset...")
    price_df = create_price_optimization_data()
    X_price, y_price, _ = prepare_data_for_modeling(price_df, 'optimal_price', ['season'])
    price_results, y_test_price = train_and_evaluate_models(X_price, y_price, "Price Optimization")
    plot_results(price_results, y_test_price, "Price Optimization")
    business_insights_analysis(price_df, 'optimal_price', "Price Optimization")
    
    # 4. Demand Forecasting
    print("\n4. Creating demand forecasting dataset...")
    demand_df = create_demand_forecasting_data()
    X_demand, y_demand, _ = prepare_data_for_modeling(demand_df, 'demand')
    demand_results, y_test_demand = train_and_evaluate_models(X_demand, y_demand, "Demand Forecasting")
    plot_results(demand_results, y_test_demand, "Demand Forecasting")
    business_insights_analysis(demand_df, 'demand', "Demand Forecasting")
    
    # 5. Employee Performance Prediction
    print("\n5. Creating employee performance dataset...")
    performance_df = create_employee_performance_data()
    X_performance, y_performance, _ = prepare_data_for_modeling(performance_df, 'performance_score', ['education_level'])
    performance_results, y_test_performance = train_and_evaluate_models(X_performance, y_performance, "Employee Performance")
    plot_results(performance_results, y_test_performance, "Employee Performance")
    business_insights_analysis(performance_df, 'performance_score', "Employee Performance")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF BEST MODELS ACROSS BUSINESS PROBLEMS")
    print(f"{'='*80}")
    
    all_results = {
        'Sales Prediction': sales_results,
        'Customer Lifetime Value': clv_results,
        'Price Optimization': price_results,
        'Demand Forecasting': demand_results,
        'Employee Performance': performance_results
    }
    
    for problem, results in all_results.items():
        best_model = max(results.items(), key=lambda x: x[1]['R2'])
        print(f"\n{problem}:")
        print(f"  Best Model: {best_model[0]}")
        print(f"  R² Score: {best_model[1]['R2']:.4f}")
        print(f"  RMSE: {best_model[1]['RMSE']:.4f}")
    
    print(f"\n{'='*80}")
    print("=== BUSINESS REGRESSION EXAMPLES COMPLETE! ===")
    print("="*80)

if __name__ == "__main__":
    main() 