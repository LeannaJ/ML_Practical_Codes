"""
Ensemble Methods Comparison
===========================

Comprehensive comparison of advanced ensemble methods:
- XGBoost
- LightGBM  
- Random Forest
- CatBoost

Includes performance comparison, training time, memory usage, and best practices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, roc_auc_score, precision_recall_curve)
from sklearn.datasets import make_classification, make_regression, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import ensemble libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

print("=== Ensemble Methods Comparison ===")

# 1. Generate Synthetic Data
print("\nGenerating synthetic datasets...")

def generate_classification_data(n_samples=2000, n_features=30, n_classes=2):
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=5,
        n_repeated=5,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def generate_regression_data(n_samples=2000, n_features=30):
    """Generate synthetic regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_targets=1,
        noise=0.1,
        random_state=42
    )
    return X, y

# Generate datasets
X_clf, y_clf = generate_classification_data()
X_reg, y_reg = generate_regression_data()

print(f"Classification data shape: {X_clf.shape}")
print(f"Regression data shape: {X_reg.shape}")

# 2. Data Preparation
print("\nPreparing data...")

# Split classification data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 3. Model Definitions
print("\nDefining models...")

models = {}

# XGBoost
if XGBOOST_AVAILABLE:
    models['XGBoost'] = {
        'classifier': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'regressor': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
    }

# LightGBM
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = {
        'classifier': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        ),
        'regressor': lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
    }

# Random Forest
models['Random Forest'] = {
    'classifier': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'regressor': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

# CatBoost
if CATBOOST_AVAILABLE:
    models['CatBoost'] = {
        'classifier': cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        ),
        'regressor': cb.CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=False
        )
    }

# 4. Training and Evaluation Functions
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def train_and_evaluate_classification(model_name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate classification model"""
    print(f"Training {model_name} classifier...")
    
    # Record start time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Record end time and memory
    end_time = time.time()
    end_memory = get_memory_usage()
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    training_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'training_time': training_time,
        'memory_usage': memory_usage,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def train_and_evaluate_regression(model_name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate regression model"""
    print(f"Training {model_name} regressor...")
    
    # Record start time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Record end time and memory
    end_time = time.time()
    end_memory = get_memory_usage()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    training_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'memory_usage': memory_usage,
        'predictions': y_pred
    }

# 5. Classification Comparison
print("\n=== Classification Comparison ===")

classification_results = {}

for model_name, model_dict in models.items():
    if 'classifier' in model_dict:
        results = train_and_evaluate_classification(
            model_name,
            model_dict['classifier'],
            X_train_clf_scaled,
            X_test_clf_scaled,
            y_train_clf,
            y_test_clf
        )
        classification_results[model_name] = results

# 6. Regression Comparison
print("\n=== Regression Comparison ===")

regression_results = {}

for model_name, model_dict in models.items():
    if 'regressor' in model_dict:
        results = train_and_evaluate_regression(
            model_name,
            model_dict['regressor'],
            X_train_reg_scaled,
            X_test_reg_scaled,
            y_train_reg,
            y_test_reg
        )
        regression_results[model_name] = results

# 7. Cross-Validation Comparison
print("\n=== Cross-Validation Comparison ===")

cv_results = {}

for model_name, model_dict in models.items():
    if 'classifier' in model_dict:
        print(f"Performing cross-validation for {model_name}...")
        cv_scores = cross_val_score(
            model_dict['classifier'],
            X_clf,
            y_clf,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        cv_results[model_name] = {
            'mean_cv': cv_scores.mean(),
            'std_cv': cv_scores.std(),
            'cv_scores': cv_scores
        }

# 8. Visualization and Analysis
print("\n=== Visualization and Analysis ===")

# Create comparison DataFrames
clf_df = pd.DataFrame(classification_results).T
reg_df = pd.DataFrame(regression_results).T
cv_df = pd.DataFrame(cv_results).T

# Classification Performance Comparison
plt.figure(figsize=(15, 10))

# Accuracy comparison
plt.subplot(2, 3, 1)
bars = plt.bar(clf_df.index, clf_df['accuracy'], alpha=0.8)
plt.title('Classification Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for bar, acc in zip(bars, clf_df['accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{acc:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# AUC comparison
plt.subplot(2, 3, 2)
bars = plt.bar(clf_df.index, clf_df['auc'], alpha=0.8)
plt.title('ROC AUC Comparison')
plt.ylabel('AUC')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for bar, auc in zip(bars, clf_df['auc']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{auc:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# Training time comparison
plt.subplot(2, 3, 3)
bars = plt.bar(clf_df.index, clf_df['training_time'], alpha=0.8)
plt.title('Training Time Comparison')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45)
for bar, time_val in zip(bars, clf_df['training_time']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{time_val:.2f}s', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# Regression Performance Comparison
plt.subplot(2, 3, 4)
bars = plt.bar(reg_df.index, reg_df['r2'], alpha=0.8)
plt.title('Regression R² Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for bar, r2 in zip(bars, reg_df['r2']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{r2:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# RMSE comparison
plt.subplot(2, 3, 5)
bars = plt.bar(reg_df.index, reg_df['rmse'], alpha=0.8)
plt.title('RMSE Comparison')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
for bar, rmse in zip(bars, reg_df['rmse']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{rmse:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# Cross-validation comparison
plt.subplot(2, 3, 6)
bars = plt.bar(cv_df.index, cv_df['mean_cv'], alpha=0.8)
plt.title('Cross-Validation Accuracy')
plt.ylabel('Mean CV Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
for bar, cv_acc in zip(bars, cv_df['mean_cv']):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{cv_acc:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Detailed Performance Analysis
print("\n=== Detailed Performance Analysis ===")

# Print classification results
print("\nClassification Results:")
print("=" * 80)
print(f"{'Model':<15} {'Accuracy':<10} {'AUC':<10} {'Time(s)':<10} {'Memory(MB)':<12}")
print("-" * 80)
for model_name, results in classification_results.items():
    print(f"{model_name:<15} {results['accuracy']:<10.4f} {results['auc']:<10.4f} "
          f"{results['training_time']:<10.2f} {results['memory_usage']:<12.2f}")

# Print regression results
print("\nRegression Results:")
print("=" * 80)
print(f"{'Model':<15} {'R²':<10} {'RMSE':<10} {'Time(s)':<10} {'Memory(MB)':<12}")
print("-" * 80)
for model_name, results in regression_results.items():
    print(f"{model_name:<15} {results['r2']:<10.4f} {results['rmse']:<10.4f} "
          f"{results['training_time']:<10.2f} {results['memory_usage']:<12.2f}")

# Print cross-validation results
print("\nCross-Validation Results:")
print("=" * 80)
print(f"{'Model':<15} {'Mean CV':<10} {'Std CV':<10}")
print("-" * 80)
for model_name, results in cv_results.items():
    print(f"{model_name:<15} {results['mean_cv']:<10.4f} {results['std_cv']:<10.4f}")

# 10. Feature Importance Comparison
print("\n=== Feature Importance Comparison ===")

# Get feature importance for all models
importance_comparison = {}

for model_name, model_dict in models.items():
    if 'classifier' in model_dict:
        model = model_dict['classifier']
        if hasattr(model, 'feature_importances_'):
            importance_comparison[model_name] = model.feature_importances_

# Plot feature importance comparison
if importance_comparison:
    plt.figure(figsize=(15, 8))
    
    # Get top 10 features from the best model
    best_model = max(classification_results.items(), key=lambda x: x[1]['accuracy'])[0]
    top_features_idx = np.argsort(importance_comparison[best_model])[-10:]
    
    # Plot feature importance for top features
    for i, (model_name, importance) in enumerate(importance_comparison.items()):
        plt.subplot(2, 2, i+1)
        plt.barh(range(len(top_features_idx)), importance[top_features_idx])
        plt.yticks(range(len(top_features_idx)), [f'Feature_{j}' for j in top_features_idx])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top 10 Features')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 11. Precision-Recall Curves
print("\n=== Precision-Recall Curves ===")

plt.figure(figsize=(10, 6))
for model_name, results in classification_results.items():
    precision, recall, _ = precision_recall_curve(y_test_clf, results['probabilities'])
    plt.plot(recall, precision, label=model_name, linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 12. Model Ranking and Recommendations
print("\n=== Model Ranking and Recommendations ===")

# Rank models by different metrics
def rank_models(results_dict, metric, higher_better=True):
    """Rank models by a specific metric"""
    sorted_models = sorted(results_dict.items(), 
                          key=lambda x: x[1][metric], 
                          reverse=higher_better)
    return sorted_models

print("\nClassification Model Rankings:")
print("=" * 50)

# Accuracy ranking
accuracy_ranking = rank_models(classification_results, 'accuracy')
print(f"\nAccuracy Ranking:")
for i, (model, results) in enumerate(accuracy_ranking, 1):
    print(f"{i}. {model}: {results['accuracy']:.4f}")

# AUC ranking
auc_ranking = rank_models(classification_results, 'auc')
print(f"\nAUC Ranking:")
for i, (model, results) in enumerate(auc_ranking, 1):
    print(f"{i}. {model}: {results['auc']:.4f}")

# Speed ranking (faster is better)
speed_ranking = rank_models(classification_results, 'training_time', higher_better=False)
print(f"\nSpeed Ranking (Fastest First):")
for i, (model, results) in enumerate(speed_ranking, 1):
    print(f"{i}. {model}: {results['training_time']:.2f}s")

print("\nRegression Model Rankings:")
print("=" * 50)

# R² ranking
r2_ranking = rank_models(regression_results, 'r2')
print(f"\nR² Ranking:")
for i, (model, results) in enumerate(r2_ranking, 1):
    print(f"{i}. {model}: {results['r2']:.4f}")

# RMSE ranking (lower is better)
rmse_ranking = rank_models(regression_results, 'rmse', higher_better=False)
print(f"\nRMSE Ranking (Best First):")
for i, (model, results) in enumerate(rmse_ranking, 1):
    print(f"{i}. {model}: {results['rmse']:.4f}")

# 13. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("\nKey Findings:")
print("=" * 50)

# Best performing models
best_clf = accuracy_ranking[0][0]
best_reg = r2_ranking[0][0]
fastest = speed_ranking[0][0]

print(f"Best Classification Model: {best_clf}")
print(f"Best Regression Model: {best_reg}")
print(f"Fastest Model: {fastest}")

print("\nModel-Specific Recommendations:")
print("=" * 50)

recommendations = {
    'XGBoost': {
        'strengths': ['Excellent performance', 'Rich feature importance', 'GPU support'],
        'best_for': ['Structured data', 'Competitions', 'Production systems'],
        'tuning': ['learning_rate', 'max_depth', 'n_estimators', 'subsample']
    },
    'LightGBM': {
        'strengths': ['Fast training', 'Low memory usage', 'Good accuracy'],
        'best_for': ['Large datasets', 'Quick prototyping', 'Memory-constrained environments'],
        'tuning': ['num_leaves', 'learning_rate', 'feature_fraction', 'bagging_fraction']
    },
    'Random Forest': {
        'strengths': ['Robust', 'No overfitting', 'Easy to interpret'],
        'best_for': ['Small to medium datasets', 'Interpretability', 'Baseline models'],
        'tuning': ['n_estimators', 'max_depth', 'min_samples_split', 'max_features']
    },
    'CatBoost': {
        'strengths': ['Categorical features', 'Ordered boosting', 'Robust'],
        'best_for': ['Categorical data', 'Tabular data', 'Production systems'],
        'tuning': ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg']
    }
}

for model_name, rec in recommendations.items():
    print(f"\n{model_name}:")
    print(f"  Strengths: {', '.join(rec['strengths'])}")
    print(f"  Best for: {', '.join(rec['best_for'])}")
    print(f"  Key parameters: {', '.join(rec['tuning'])}")

print("\nGeneral Best Practices:")
print("=" * 50)
print("1. Start with Random Forest for baseline performance")
print("2. Use LightGBM for large datasets or quick prototyping")
print("3. Use XGBoost for maximum performance (with proper tuning)")
print("4. Use CatBoost for datasets with many categorical features")
print("5. Always use cross-validation for robust evaluation")
print("6. Monitor training time and memory usage")
print("7. Tune hyperparameters systematically")
print("8. Consider ensemble methods for final predictions")
print("9. Use appropriate evaluation metrics for your problem")
print("10. Scale features for better performance")

print("\nWhen to Use Each Algorithm:")
print("=" * 50)
print("XGBoost: When you need maximum performance and have time for tuning")
print("LightGBM: When you need fast training and have large datasets")
print("Random Forest: When you need interpretability and robustness")
print("CatBoost: When you have many categorical features")
print("Ensemble: When you want to combine the strengths of multiple algorithms") 