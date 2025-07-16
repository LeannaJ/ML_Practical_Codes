"""
Regression Algorithms Comparison
===============================

This script demonstrates and compares various regression algorithms including:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest Regression
- Gradient Boosting Regression
- Support Vector Regression
- Neural Network Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_datasets():
    """Create multiple sample datasets for regression"""
    n_samples = 1000
    
    # Dataset 1: Linear relationship
    np.random.seed(42)
    X1_linear = np.random.randn(n_samples, 5)
    y1_linear = 2 * X1_linear[:, 0] + 1.5 * X1_linear[:, 1] - 0.5 * X1_linear[:, 2] + np.random.normal(0, 0.1, n_samples)
    
    # Dataset 2: Non-linear relationship
    X2_nonlinear = np.random.randn(n_samples, 3)
    y2_nonlinear = X2_nonlinear[:, 0]**2 + 2 * X2_nonlinear[:, 1] + np.sin(X2_nonlinear[:, 2]) + np.random.normal(0, 0.2, n_samples)
    
    # Dataset 3: High-dimensional dataset
    X3_highdim = np.random.randn(n_samples, 20)
    y3_highdim = np.sum(X3_highdim[:, :5], axis=1) + np.random.normal(0, 0.3, n_samples)
    
    # Dataset 4: Real-world like dataset (house prices)
    size = np.random.normal(1500, 500, n_samples)
    bedrooms = np.random.poisson(3, n_samples)
    age = np.random.exponential(20, n_samples)
    distance_to_city = np.random.exponential(10, n_samples)
    
    X4_house = np.column_stack([size, bedrooms, age, distance_to_city])
    y4_house = 200000 + 100 * size - 10000 * age - 5000 * distance_to_city + 15000 * bedrooms + np.random.normal(0, 10000, n_samples)
    
    datasets = {
        'Linear': (X1_linear, y1_linear),
        'Non-linear': (X2_nonlinear, y2_nonlinear),
        'High-dimensional': (X3_highdim, y3_highdim),
        'House Prices': (X4_house, y4_house)
    }
    
    return datasets

def get_regression_models():
    """Get various regression models for comparison"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    return models

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a single model"""
    print(f"\nEvaluating {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_R2_mean': cv_mean,
        'CV_R2_std': cv_std
    }
    
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  CV R²: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    return results, y_pred

def compare_models_on_dataset(X, y, dataset_name):
    """Compare all models on a specific dataset"""
    print(f"\n{'='*60}")
    print(f"COMPARING MODELS ON {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Get models
    models = get_regression_models()
    
    # Evaluate all models
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # Use scaled data for models that benefit from scaling
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Support Vector Regression', 'Neural Network']:
            result, pred = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
        else:
            result, pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        
        results[name] = result
        predictions[name] = pred
    
    return results, predictions, y_test

def plot_model_comparison(results, dataset_name):
    """Plot comparison of models"""
    print(f"\nPlotting results for {dataset_name} dataset...")
    
    # Extract metrics
    models = list(results.keys())
    r2_scores = [results[model]['R2'] for model in models]
    rmse_scores = [results[model]['RMSE'] for model in models]
    cv_scores = [results[model]['CV_R2_mean'] for model in models]
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # R² Score comparison
    bars1 = axes[0].bar(models, r2_scores)
    axes[0].set_title(f'R² Score Comparison - {dataset_name}')
    axes[0].set_ylabel('R² Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    
    # RMSE comparison
    bars2 = axes[1].bar(models, rmse_scores)
    axes[1].set_title(f'RMSE Comparison - {dataset_name}')
    axes[1].set_ylabel('RMSE')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    
    # Cross-validation R² comparison
    bars3 = axes[2].bar(models, cv_scores)
    axes[2].set_title(f'Cross-validation R² Comparison - {dataset_name}')
    axes[2].set_ylabel('CV R² Score')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars3, cv_scores):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print ranking
    print(f"\nModel Ranking for {dataset_name} dataset (by R² Score):")
    ranking = sorted(results.items(), key=lambda x: x[1]['R2'], reverse=True)
    for i, (model, metrics) in enumerate(ranking, 1):
        print(f"{i}. {model}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")

def plot_predictions_vs_actual(predictions, y_test, dataset_name):
    """Plot predictions vs actual values"""
    print(f"\nPlotting predictions vs actual for {dataset_name} dataset...")
    
    # Select top 3 models for visualization
    models_to_plot = list(predictions.keys())[:3]
    
    fig, axes = plt.subplots(1, len(models_to_plot), figsize=(15, 5))
    if len(models_to_plot) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(models_to_plot):
        y_pred = predictions[model_name]
        
        axes[i].scatter(y_test, y_pred, alpha=0.6)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].set_title(f'{model_name}\nR² = {r2_score(y_test, y_pred):.3f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def hyperparameter_tuning_example(X, y, dataset_name):
    """Demonstrate hyperparameter tuning"""
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING EXAMPLE - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Example: Tune Random Forest
    print("Tuning Random Forest hyperparameters...")
    
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Best model performance on test set:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    return best_model, grid_search.best_params_

def feature_importance_analysis(X, y, dataset_name):
    """Analyze feature importance for tree-based models"""
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 most important features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {dataset_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

def main():
    """Main function to run the complete regression comparison"""
    print("=== REGRESSION ALGORITHMS COMPARISON ===\n")
    
    # Create datasets
    print("1. Creating sample datasets...")
    datasets = create_sample_datasets()
    
    # Compare models on each dataset
    all_results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        
        # Compare models
        results, predictions, y_test = compare_models_on_dataset(X, y, dataset_name)
        all_results[dataset_name] = results
        
        # Plot comparisons
        plot_model_comparison(results, dataset_name)
        
        # Plot predictions vs actual
        plot_predictions_vs_actual(predictions, y_test, dataset_name)
        
        # Feature importance analysis (for datasets with reasonable number of features)
        if X.shape[1] <= 20:
            feature_importance_analysis(X, y, dataset_name)
        
        # Hyperparameter tuning example (for one dataset)
        if dataset_name == 'House Prices':
            hyperparameter_tuning_example(X, y, dataset_name)
    
    # Summary of best models across datasets
    print(f"\n{'='*80}")
    print("SUMMARY OF BEST MODELS ACROSS DATASETS")
    print(f"{'='*80}")
    
    for dataset_name, results in all_results.items():
        best_model = max(results.items(), key=lambda x: x[1]['R2'])
        print(f"\n{dataset_name}:")
        print(f"  Best Model: {best_model[0]}")
        print(f"  R² Score: {best_model[1]['R2']:.4f}")
        print(f"  RMSE: {best_model[1]['RMSE']:.4f}")
    
    print(f"\n{'='*80}")
    print("=== REGRESSION COMPARISON COMPLETE! ===")
    print("="*80)

if __name__ == "__main__":
    main() 