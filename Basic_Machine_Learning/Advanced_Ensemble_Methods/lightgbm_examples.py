"""
LightGBM Examples
=================

- Classification with LightGBM
- Regression with LightGBM
- Hyperparameter Tuning
- Feature Importance Analysis
- Cross-validation
- Early Stopping
- Custom Evaluation Metrics
- Advanced Techniques (DART, GPU acceleration)
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
import warnings
warnings.filterwarnings('ignore')

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

# 1. Generate Synthetic Data
print("=== LightGBM Examples ===")

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

# 2. Basic LightGBM Classification
print("\n=== Basic LightGBM Classification ===")

if LIGHTGBM_AVAILABLE:
    # Split classification data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_clf, label=y_train_clf)
    valid_data = lgb.Dataset(X_test_clf, label=y_test_clf, reference=train_data)
    
    # Set parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Train model
    print("Training LightGBM classification model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test_clf, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Evaluate
    accuracy = accuracy_score(y_test_clf, y_pred)
    auc = roc_auc_score(y_test_clf, y_pred_proba)
    
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_clf, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_clf, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - LightGBM Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 3. Basic LightGBM Regression
print("\n=== Basic LightGBM Regression ===")

if LIGHTGBM_AVAILABLE:
    # Split regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Create LightGBM datasets
    train_data_reg = lgb.Dataset(X_train_reg, label=y_train_reg)
    valid_data_reg = lgb.Dataset(X_test_reg, label=y_test_reg, reference=train_data_reg)
    
    # Set parameters for regression
    params_reg = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Train model
    print("Training LightGBM regression model...")
    model_reg = lgb.train(
        params_reg,
        train_data_reg,
        valid_sets=[valid_data_reg],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    # Make predictions
    y_pred_reg = model_reg.predict(X_test_reg, num_iteration=model_reg.best_iteration)
    
    # Evaluate
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('LightGBM Regression: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    plt.show()

# 4. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

if LIGHTGBM_AVAILABLE:
    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    
    # Create feature names
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
    plt.xlabel('Feature Importance (Gain)')
    plt.title('LightGBM Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    for i, (feature, imp) in enumerate(importance_df.head(10).values):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    # Feature importance by type
    importance_split = model.feature_importance(importance_type='split')
    importance_gain = model.feature_importance(importance_type='gain')
    
    print(f"\nFeature importance by type:")
    print(f"Split importance: {len([x for x in importance_split if x > 0])} features")
    print(f"Gain importance: {len([x for x in importance_gain if x > 0])} features")

# 5. Hyperparameter Tuning
print("\n=== Hyperparameter Tuning ===")

if LIGHTGBM_AVAILABLE:
    # Grid Search for hyperparameter tuning
    param_grid = {
        'num_leaves': [15, 31, 63],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'feature_fraction': [0.8, 0.9, 1.0],
        'bagging_fraction': [0.8, 0.9, 1.0]
    }
    
    # Create LightGBM classifier for sklearn API
    lgb_clf = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        boosting_type='gbdt',
        random_state=42,
        verbose=-1
    )
    
    # Grid search
    print("Performing Grid Search...")
    grid_search = GridSearchCV(
        lgb_clf,
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
        lgb_clf,
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

if LIGHTGBM_AVAILABLE:
    # Cross-validation with LightGBM
    lgb_cv = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(lgb_cv, X_clf, y_clf, cv=5, scoring='accuracy')
    
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
    plt.title('LightGBM Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 7. Advanced LightGBM Techniques
print("\n=== Advanced LightGBM Techniques ===")

if LIGHTGBM_AVAILABLE:
    # DART (Dropouts meet Multiple Additive Regression Trees)
    print("Training DART model...")
    dart_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'dart',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'verbose': -1,
        'random_state': 42
    }
    
    dart_model = lgb.train(
        dart_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    dart_pred = dart_model.predict(X_test_clf, num_iteration=dart_model.best_iteration)
    dart_accuracy = accuracy_score(y_test_clf, (dart_pred > 0.5).astype(int))
    print(f"DART Accuracy: {dart_accuracy:.4f}")
    
    # GOSS (Gradient-based One-Side Sampling)
    print("\nTraining GOSS model...")
    goss_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'top_rate': 0.2,
        'other_rate': 0.1,
        'verbose': -1,
        'random_state': 42
    }
    
    goss_model = lgb.train(
        goss_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    goss_pred = goss_model.predict(X_test_clf, num_iteration=goss_model.best_iteration)
    goss_accuracy = accuracy_score(y_test_clf, (goss_pred > 0.5).astype(int))
    print(f"GOSS Accuracy: {goss_accuracy:.4f}")
    
    # Custom evaluation metric
    def custom_metric(y_true, y_pred):
        """Custom evaluation metric"""
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
        return 'custom_metric', np.mean(np.abs(y_true - y_pred)), False
    
    # Train with custom metric
    print("\nTraining with custom evaluation metric...")
    custom_params = params.copy()
    
    custom_model = lgb.train(
        custom_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        feval=custom_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    custom_pred = custom_model.predict(X_test_clf, num_iteration=custom_model.best_iteration)
    custom_accuracy = accuracy_score(y_test_clf, (custom_pred > 0.5).astype(int))
    print(f"Custom Metric Model Accuracy: {custom_accuracy:.4f}")

# 8. Model Interpretation and Visualization
print("\n=== Model Interpretation and Visualization ===")

if LIGHTGBM_AVAILABLE:
    # Learning curves
    evals_result = model.best_score
    
    # Plot feature importance comparison
    plt.figure(figsize=(12, 5))
    
    # Feature importance by different types
    importance_split = model.feature_importance(importance_type='split')
    importance_gain = model.feature_importance(importance_type='gain')
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(importance_split)), importance_split, alpha=0.7, label='Split')
    plt.bar(range(len(importance_gain)), importance_gain, alpha=0.7, label='Gain')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Top features comparison
    plt.subplot(1, 2, 2)
    top_features = 10
    top_split_idx = np.argsort(importance_split)[-top_features:]
    top_gain_idx = np.argsort(importance_gain)[-top_features:]
    
    plt.bar(range(top_features), importance_split[top_split_idx], alpha=0.7, label='Split')
    plt.bar(range(top_features), importance_gain[top_gain_idx], alpha=0.7, label='Gain')
    plt.xlabel('Top Features')
    plt.ylabel('Importance')
    plt.title('Top 10 Features Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Precision-Recall curve
    y_pred_proba = model.predict(X_test_clf, num_iteration=model.best_iteration)
    precision, recall, _ = precision_recall_curve(y_test_clf, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='LightGBM')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 9. Real-world Dataset Example
print("\n=== Real-world Dataset Example ===")

if LIGHTGBM_AVAILABLE:
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
        
        # Train LightGBM
        cancer_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            random_state=42,
            verbose=-1
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

if LIGHTGBM_AVAILABLE:
    # Compare different LightGBM configurations
    models_comparison = {
        'GBDT': model,
        'DART': dart_model,
        'GOSS': goss_model
    }
    
    comparison_results = {}
    
    for name, model_obj in models_comparison.items():
        if name == 'GBDT':
            pred = model_obj.predict(X_test_clf, num_iteration=model_obj.best_iteration)
        else:
            pred = model_obj.predict(X_test_clf, num_iteration=model_obj.best_iteration)
        
        acc = accuracy_score(y_test_clf, (pred > 0.5).astype(int))
        comparison_results[name] = acc
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_results.keys(), comparison_results.values(), alpha=0.8)
    plt.title('LightGBM Model Comparison')
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

# 11. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("LightGBM Performance Summary:")
if LIGHTGBM_AVAILABLE:
    print(f"1. Classification Accuracy: {accuracy:.4f}")
    print(f"2. Regression R² Score: {r2:.4f}")
    print(f"3. Cross-validation Mean Score: {cv_scores.mean():.4f}")

print(f"\nKey LightGBM Advantages:")
print(f"- Faster training speed compared to XGBoost")
print(f"- Lower memory usage")
print(f"- Better accuracy on many datasets")
print(f"- Native support for categorical features")
print(f"- GPU acceleration support")
print(f"- Leaf-wise tree growth")

print(f"\nBest Practices:")
print(f"1. Start with default parameters and tune gradually")
print(f"2. Use early stopping to prevent overfitting")
print(f"3. Tune num_leaves and learning_rate together")
print(f"4. Use cross-validation for robust evaluation")
print(f"5. Monitor feature importance for feature selection")
print(f"6. Consider DART for better generalization")
print(f"7. Use GOSS for large datasets")
print(f"8. Handle categorical features properly")
print(f"9. Use appropriate evaluation metrics")
print(f"10. Scale features for better performance")

print(f"\nCommon Parameters to Tune:")
print(f"- num_leaves: Number of leaves (15-255)")
print(f"- learning_rate: Step size (0.01-0.3)")
print(f"- n_estimators: Number of trees (50-1000)")
print(f"- feature_fraction: Feature sampling (0.6-1.0)")
print(f"- bagging_fraction: Row sampling (0.6-1.0)")
print(f"- reg_alpha: L1 regularization")
print(f"- reg_lambda: L2 regularization")
print(f"- min_child_samples: Minimum samples per leaf")
print(f"- min_split_gain: Minimum gain for split")

print(f"\nAdvanced Techniques:")
print(f"- DART: Dropouts meet Multiple Additive Regression Trees")
print(f"- GOSS: Gradient-based One-Side Sampling")
print(f"- EFB: Exclusive Feature Bundling")
print(f"- GPU acceleration for large datasets")
print(f"- Categorical feature handling")
print(f"- Custom evaluation metrics") 