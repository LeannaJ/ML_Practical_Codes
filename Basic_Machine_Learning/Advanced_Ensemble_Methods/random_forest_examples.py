"""
Random Forest Examples
======================

- Classification with Random Forest
- Regression with Random Forest
- Hyperparameter Tuning
- Feature Importance Analysis
- Cross-validation
- Out-of-bag (OOB) estimation
- Advanced Techniques (Extra Trees, Isolation Forest)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           mean_squared_error, r2_score, roc_auc_score, precision_recall_curve)
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_boston
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# 1. Generate Synthetic Data
print("=== Random Forest Examples ===")

def generate_classification_data(n_samples=1000, n_features=20, n_classes=2):
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=3,
        n_repeated=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, y

def generate_regression_data(n_samples=1000, n_features=20):
    """Generate synthetic regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_targets=1,
        noise=0.1,
        random_state=42
    )
    return X, y

# Generate datasets
print("Generating synthetic datasets...")
X_clf, y_clf = generate_classification_data()
X_reg, y_reg = generate_regression_data()

print(f"Classification data shape: {X_clf.shape}")
print(f"Regression data shape: {X_reg.shape}")

# 2. Basic Random Forest Classification
print("\n=== Basic Random Forest Classification ===")

# Split classification data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Create and train Random Forest classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest classification model...")
rf_clf.fit(X_train_clf, y_train_clf)

# Make predictions
y_pred_clf = rf_clf.predict(X_test_clf)
y_pred_proba_clf = rf_clf.predict_proba(X_test_clf)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test_clf, y_pred_clf)
auc = roc_auc_score(y_test_clf, y_pred_proba_clf)
oob_score = rf_clf.oob_score_

print(f"Classification Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {auc:.4f}")
print(f"Out-of-bag Score: {oob_score:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test_clf, y_pred_clf))

# Confusion matrix
cm = confusion_matrix(y_test_clf, y_pred_clf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 3. Basic Random Forest Regression
print("\n=== Basic Random Forest Regression ===")

# Split regression data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Create and train Random Forest regressor
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest regression model...")
rf_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = rf_reg.predict(X_test_reg)

# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)
oob_score_reg = rf_reg.oob_score_

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Out-of-bag Score: {oob_score_reg:.4f}")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression: Predicted vs Actual')
plt.grid(True, alpha=0.3)
plt.show()

# 4. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# Get feature importance
importance = rf_clf.feature_importances_
feature_names = [f'feature_{i}' for i in range(len(importance))]

# Convert to DataFrame for easier plotting
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print top features
print("\nTop 10 Most Important Features:")
for i, (feature, imp) in enumerate(importance_df.head(10).values):
    print(f"{i+1}. {feature}: {imp:.4f}")

# Feature importance comparison between classification and regression
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(rf_clf.feature_importances_)), rf_clf.feature_importances_)
plt.title('Classification Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(len(rf_reg.feature_importances_)), rf_reg.feature_importances_)
plt.title('Regression Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Hyperparameter Tuning
print("\n=== Hyperparameter Tuning ===")

# Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search
print("Performing Grid Search...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_clf, y_train_clf)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Random Search (faster alternative)
print("\nPerforming Random Search...")
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train_clf, y_train_clf)

print(f"Best parameters (Random Search): {random_search.best_params_}")
print(f"Best cross-validation score (Random Search): {random_search.best_score_:.4f}")

# 6. Cross-Validation
print("\n=== Cross-Validation ===")

# Cross-validation with Random Forest
rf_cv = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# Perform cross-validation
cv_scores = cross_val_score(rf_cv, X_clf, y_clf, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Plot CV results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), cv_scores, 'bo-', label='CV Scores')
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.fill_between(range(1, 6), 
                 cv_scores.mean() - cv_scores.std(),
                 cv_scores.mean() + cv_scores.std(),
                 alpha=0.2, color='r', label=f'±1 std: {cv_scores.std():.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Random Forest Cross-Validation Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. Advanced Random Forest Techniques
print("\n=== Advanced Random Forest Techniques ===")

# Extra Trees (Extremely Randomized Trees)
print("Training Extra Trees classifier...")
et_clf = ExtraTreesClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

et_clf.fit(X_train_clf, y_train_clf)
et_pred = et_clf.predict(X_test_clf)
et_accuracy = accuracy_score(y_test_clf, et_pred)
print(f"Extra Trees Accuracy: {et_accuracy:.4f}")

# Extra Trees Regressor
print("\nTraining Extra Trees regressor...")
et_reg = ExtraTreesRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

et_reg.fit(X_train_reg, y_train_reg)
et_pred_reg = et_reg.predict(X_test_reg)
et_r2 = r2_score(y_test_reg, et_pred_reg)
print(f"Extra Trees R² Score: {et_r2:.4f}")

# Isolation Forest for Anomaly Detection
print("\nTraining Isolation Forest for anomaly detection...")
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42,
    n_jobs=-1
)

# Fit on training data
iso_forest.fit(X_train_clf)

# Predict anomalies (-1 for anomalies, 1 for normal)
anomaly_scores = iso_forest.predict(X_test_clf)
n_anomalies = np.sum(anomaly_scores == -1)
print(f"Number of detected anomalies: {n_anomalies}")

# 8. Model Interpretation and Visualization
print("\n=== Model Interpretation and Visualization ===")

# Learning curves (number of trees vs performance)
n_trees_range = [10, 25, 50, 100, 200]
train_scores = []
test_scores = []

for n_trees in n_trees_range:
    rf_temp = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_temp.fit(X_train_clf, y_train_clf)
    train_scores.append(rf_temp.score(X_train_clf, y_train_clf))
    test_scores.append(rf_temp.score(X_test_clf, y_test_clf))

plt.figure(figsize=(12, 5))

# Learning curves
plt.subplot(1, 2, 1)
plt.plot(n_trees_range, train_scores, 'bo-', label='Training Score')
plt.plot(n_trees_range, test_scores, 'ro-', label='Test Score')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Random Forest Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance comparison
plt.subplot(1, 2, 2)
plt.bar(range(len(rf_clf.feature_importances_)), rf_clf.feature_importances_, alpha=0.7, label='Random Forest')
plt.bar(range(len(et_clf.feature_importances_)), et_clf.feature_importances_, alpha=0.7, label='Extra Trees')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Precision-Recall curve
y_pred_proba_clf = rf_clf.predict_proba(X_test_clf)[:, 1]
precision, recall, _ = precision_recall_curve(y_test_clf, y_pred_proba_clf)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Random Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 9. Real-world Dataset Example
print("\n=== Real-world Dataset Example ===")

# Load breast cancer dataset
try:
    cancer = load_breast_cancer()
    X_cancer = cancer.data
    y_cancer = cancer.target
    
    # Split data
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_cancer_scaled = scaler.fit_transform(X_train_cancer)
    X_test_cancer_scaled = scaler.transform(X_test_cancer)
    
    # Train Random Forest
    cancer_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    cancer_model.fit(X_train_cancer_scaled, y_train_cancer)
    
    # Predictions
    y_pred_cancer = cancer_model.predict(X_test_cancer_scaled)
    y_pred_proba_cancer = cancer_model.predict_proba(X_test_cancer_scaled)[:, 1]
    
    # Evaluate
    cancer_accuracy = accuracy_score(y_test_cancer, y_pred_cancer)
    cancer_auc = roc_auc_score(y_test_cancer, y_pred_proba_cancer)
    
    print(f"Breast Cancer Dataset Results:")
    print(f"Accuracy: {cancer_accuracy:.4f}")
    print(f"ROC AUC: {cancer_auc:.4f}")
    
    # Feature importance for cancer dataset
    cancer_importance = cancer_model.feature_importances_
    feature_names = cancer.feature_names
    
    # Plot top features
    top_features_idx = np.argsort(cancer_importance)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features_idx)), cancer_importance[top_features_idx])
    plt.yticks(range(len(top_features_idx)), [feature_names[i] for i in top_features_idx])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Important Features - Breast Cancer Dataset')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error loading breast cancer dataset: {e}")

# 10. Model Comparison
print("\n=== Model Comparison ===")

# Compare different Random Forest configurations
models_comparison = {
    'Random Forest': rf_clf,
    'Extra Trees': et_clf
}

comparison_results = {}

for name, model_obj in models_comparison.items():
    pred = model_obj.predict(X_test_clf)
    acc = accuracy_score(y_test_clf, pred)
    comparison_results[name] = acc

# Plot comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(comparison_results.keys(), comparison_results.values(), alpha=0.8)
plt.title('Random Forest Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, comparison_results.values()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Regression model comparison
reg_models_comparison = {
    'Random Forest': rf_reg,
    'Extra Trees': et_reg
}

reg_comparison_results = {}

for name, model_obj in reg_models_comparison.items():
    pred = model_obj.predict(X_test_reg)
    r2 = r2_score(y_test_reg, pred)
    reg_comparison_results[name] = r2

# Plot regression comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(reg_comparison_results.keys(), reg_comparison_results.values(), alpha=0.8)
plt.title('Random Forest Regression Model Comparison')
plt.ylabel('R² Score')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, r2 in zip(bars, reg_comparison_results.values()):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{r2:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 11. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("Random Forest Performance Summary:")
print(f"1. Classification Accuracy: {accuracy:.4f}")
print(f"2. Regression R² Score: {r2:.4f}")
print(f"3. Cross-validation Mean Score: {cv_scores.mean():.4f}")
print(f"4. Out-of-bag Score (Classification): {oob_score:.4f}")
print(f"5. Out-of-bag Score (Regression): {oob_score_reg:.4f}")

print(f"\nKey Random Forest Advantages:")
print(f"- Handles both classification and regression")
print(f"- Provides feature importance")
print(f"- Robust to overfitting")
print(f"- Handles missing values")
print(f"- No feature scaling required")
print(f"- Out-of-bag estimation")
print(f"- Parallelizable training")

print(f"\nBest Practices:")
print(f"1. Start with default parameters and tune gradually")
print(f"2. Use out-of-bag score for validation")
print(f"3. Tune n_estimators and max_depth together")
print(f"4. Use cross-validation for robust evaluation")
print(f"5. Monitor feature importance for feature selection")
print(f"6. Consider Extra Trees for better generalization")
print(f"7. Use appropriate evaluation metrics")
print(f"8. Handle class imbalance with class_weight")
print(f"9. Use n_jobs=-1 for parallel processing")
print(f"10. Consider ensemble with other algorithms")

print(f"\nCommon Parameters to Tune:")
print(f"- n_estimators: Number of trees (50-500)")
print(f"- max_depth: Tree depth (5-20)")
print(f"- min_samples_split: Minimum samples to split (2-10)")
print(f"- min_samples_leaf: Minimum samples per leaf (1-5)")
print(f"- max_features: Feature sampling ('sqrt', 'log2', None)")
print(f"- bootstrap: Whether to use bootstrapping (True/False)")

print(f"\nAdvanced Techniques:")
print(f"- Extra Trees: Extremely Randomized Trees")
print(f"- Isolation Forest: Anomaly detection")
print(f"- Feature importance analysis")
print(f"- Out-of-bag estimation")
print(f"- Ensemble with other algorithms")
print(f"- Custom evaluation metrics") 