"""
XGBoost Examples
================

- Classification with XGBoost
- Regression with XGBoost
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

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

# 1. Generate Synthetic Data
print("=== XGBoost Examples ===")

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

# 2. Basic XGBoost Classification
print("\n=== Basic XGBoost Classification ===")

if XGBOOST_AVAILABLE:
    # Split classification data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_clf, label=y_train_clf)
    dtest = xgb.DMatrix(X_test_clf, label=y_test_clf)
    
    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Train model
    print("Training XGBoost classification model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred_proba = model.predict(dtest)
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
    plt.title('Confusion Matrix - XGBoost Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 3. Basic XGBoost Regression
print("\n=== Basic XGBoost Regression ===")

if XGBOOST_AVAILABLE:
    # Split regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Create DMatrix
    dtrain_reg = xgb.DMatrix(X_train_reg, label=y_train_reg)
    dtest_reg = xgb.DMatrix(X_test_reg, label=y_test_reg)
    
    # Set parameters for regression
    params_reg = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Train model
    print("Training XGBoost regression model...")
    model_reg = xgb.train(
        params_reg,
        dtrain_reg,
        num_boost_round=100,
        evals=[(dtrain_reg, 'train'), (dtest_reg, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Make predictions
    y_pred_reg = model_reg.predict(dtest_reg)
    
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
    plt.title('XGBoost Regression: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    plt.show()

# 4. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

if XGBOOST_AVAILABLE:
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    
    # Convert to DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (Gain)')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    for i, (feature, imp) in enumerate(importance_df.head(10).values):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    # SHAP-like feature importance (simplified)
    print("\nFeature importance by type:")
    print(f"Gain importance: {len([k for k in importance.keys() if 'gain' in k])} features")
    print(f"Cover importance: {len([k for k in importance.keys() if 'cover' in k])} features")
    print(f"Frequency importance: {len([k for k in importance.keys() if 'freq' in k])} features")

# 5. Hyperparameter Tuning
print("\n=== Hyperparameter Tuning ===")

if XGBOOST_AVAILABLE:
    # Grid Search for hyperparameter tuning
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create XGBoost classifier for sklearn API
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    # Grid search
    print("Performing Grid Search...")
    grid_search = GridSearchCV(
        xgb_clf,
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
        xgb_clf,
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

if XGBOOST_AVAILABLE:
    # Cross-validation with XGBoost
    xgb_cv = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        use_label_encoder=False
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(xgb_cv, X_clf, y_clf, cv=5, scoring='accuracy')
    
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
    plt.title('XGBoost Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 7. Advanced XGBoost Techniques
print("\n=== Advanced XGBoost Techniques ===")

if XGBOOST_AVAILABLE:
    # DART (Dropouts meet Multiple Additive Regression Trees)
    print("Training DART model...")
    dart_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'booster': 'dart',  # Use DART booster
        'sample_type': 'uniform',
        'normalize_type': 'tree',
        'rate_drop': 0.1,
        'skip_drop': 0.5
    }
    
    dart_model = xgb.train(
        dart_params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    dart_pred = dart_model.predict(dtest)
    dart_accuracy = accuracy_score(y_test_clf, (dart_pred > 0.5).astype(int))
    print(f"DART Accuracy: {dart_accuracy:.4f}")
    
    # Custom evaluation metric
    def custom_metric(predt, dtrain):
        """Custom evaluation metric"""
        y_true = dtrain.get_label()
        predt = 1.0 / (1.0 + np.exp(-predt))  # sigmoid
        return 'custom_metric', np.mean(np.abs(y_true - predt))
    
    # Train with custom metric
    print("\nTraining with custom evaluation metric...")
    custom_params = params.copy()
    custom_params['eval_metric'] = custom_metric
    
    custom_model = xgb.train(
        custom_params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    custom_pred = custom_model.predict(dtest)
    custom_accuracy = accuracy_score(y_test_clf, (custom_pred > 0.5).astype(int))
    print(f"Custom Metric Model Accuracy: {custom_accuracy:.4f}")

# 8. Model Interpretation and Visualization
print("\n=== Model Interpretation and Visualization ===")

if XGBOOST_AVAILABLE:
    # Learning curves
    results = model.evals_result()
    
    plt.figure(figsize=(12, 5))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    plt.plot(results['train']['logloss'], label='Train')
    plt.plot(results['test']['logloss'], label='Test')
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature importance comparison
    plt.subplot(1, 2, 2)
    importance_types = ['gain', 'cover', 'total_gain', 'total_cover']
    importance_data = {}
    
    for imp_type in importance_types:
        try:
            imp = model.get_score(importance_type=imp_type)
            importance_data[imp_type] = len(imp)
        except:
            importance_data[imp_type] = 0
    
    plt.bar(importance_data.keys(), importance_data.values())
    plt.xlabel('Importance Type')
    plt.ylabel('Number of Features')
    plt.title('Feature Importance by Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Precision-Recall curve
    y_pred_proba = model.predict(dtest)
    precision, recall, _ = precision_recall_curve(y_test_clf, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='XGBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 9. Real-world Dataset Example
print("\n=== Real-world Dataset Example ===")

if XGBOOST_AVAILABLE:
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
        
        # Train XGBoost
        cancer_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=4,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            use_label_encoder=False
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

# 10. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("XGBoost Performance Summary:")
if XGBOOST_AVAILABLE:
    print(f"1. Classification Accuracy: {accuracy:.4f}")
    print(f"2. Regression R² Score: {r2:.4f}")
    print(f"3. Cross-validation Mean Score: {cv_scores.mean():.4f}")

print(f"\nKey XGBoost Advantages:")
print(f"- Handles missing values automatically")
print(f"- Built-in regularization to prevent overfitting")
print(f"- Supports early stopping")
print(f"- Efficient implementation with GPU support")
print(f"- Rich feature importance analysis")
print(f"- Handles both classification and regression")

print(f"\nBest Practices:")
print(f"1. Start with default parameters and tune gradually")
print(f"2. Use early stopping to prevent overfitting")
print(f"3. Tune learning_rate and n_estimators together")
print(f"4. Use cross-validation for robust evaluation")
print(f"5. Monitor feature importance for feature selection")
print(f"6. Consider DART for better generalization")
print(f"7. Use appropriate evaluation metrics")
print(f"8. Scale features for better performance")
print(f"9. Handle class imbalance with scale_pos_weight")
print(f"10. Use GPU acceleration for large datasets")

print(f"\nCommon Parameters to Tune:")
print(f"- max_depth: Tree depth (3-10)")
print(f"- learning_rate: Step size (0.01-0.3)")
print(f"- n_estimators: Number of trees (50-1000)")
print(f"- subsample: Row sampling (0.6-1.0)")
print(f"- colsample_bytree: Column sampling (0.6-1.0)")
print(f"- reg_alpha: L1 regularization")
print(f"- reg_lambda: L2 regularization") 