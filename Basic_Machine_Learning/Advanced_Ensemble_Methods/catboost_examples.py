"""
CatBoost Examples
=================

- Classification with CatBoost
- Regression with CatBoost
- Hyperparameter Tuning
- Feature Importance Analysis
- Cross-validation
- Early Stopping
- Categorical Feature Handling
- Advanced Techniques
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

# CatBoost
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

# 1. Generate Synthetic Data
print("=== CatBoost Examples ===")

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

def generate_categorical_data(n_samples=1000, n_features=20, n_categorical=5):
    """Generate data with categorical features"""
    np.random.seed(42)
    
    # Generate numerical features
    X_num = np.random.randn(n_samples, n_features - n_categorical)
    
    # Generate categorical features
    X_cat = []
    for i in range(n_categorical):
        n_categories = np.random.randint(3, 10)
        cat_feature = np.random.randint(0, n_categories, n_samples)
        X_cat.append(cat_feature)
    
    X_cat = np.column_stack(X_cat)
    
    # Combine numerical and categorical features
    X = np.column_stack([X_num, X_cat])
    
    # Generate target (classification)
    y = np.random.randint(0, 2, n_samples)
    
    return X, y, list(range(n_features - n_categorical, n_features))

# Generate datasets
print("Generating synthetic datasets...")
X_clf, y_clf = generate_classification_data()
X_reg, y_reg = generate_regression_data()
X_cat, y_cat, cat_features = generate_categorical_data()

print(f"Classification data shape: {X_clf.shape}")
print(f"Regression data shape: {X_reg.shape}")
print(f"Categorical data shape: {X_cat.shape}")
print(f"Categorical features: {cat_features}")

# 2. Basic CatBoost Classification
print("\n=== Basic CatBoost Classification ===")

if CATBOOST_AVAILABLE:
    # Split classification data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    # Create CatBoost datasets
    train_data = cb.Pool(X_train_clf, label=y_train_clf)
    test_data = cb.Pool(X_test_clf, label=y_test_clf)
    
    # Set parameters
    params = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': False
    }
    
    # Train model
    print("Training CatBoost classification model...")
    model = cb.CatBoostClassifier(**params)
    model.fit(train_data, eval_set=test_data, early_stopping_rounds=10, verbose=False)
    
    # Make predictions
    y_pred_clf = model.predict(X_test_clf)
    y_pred_proba_clf = model.predict_proba(X_test_clf)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    auc = roc_auc_score(y_test_clf, y_pred_proba_clf)
    
    print(f"Classification Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_clf, y_pred_clf))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - CatBoost Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# 3. Basic CatBoost Regression
print("\n=== Basic CatBoost Regression ===")

if CATBOOST_AVAILABLE:
    # Split regression data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # Create CatBoost datasets
    train_data_reg = cb.Pool(X_train_reg, label=y_train_reg)
    test_data_reg = cb.Pool(X_test_reg, label=y_test_reg)
    
    # Set parameters for regression
    params_reg = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': False
    }
    
    # Train model
    print("Training CatBoost regression model...")
    model_reg = cb.CatBoostRegressor(**params_reg)
    model_reg.fit(train_data_reg, eval_set=test_data_reg, early_stopping_rounds=10, verbose=False)
    
    # Make predictions
    y_pred_reg = model_reg.predict(X_test_reg)
    
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
    plt.title('CatBoost Regression: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    plt.show()

# 4. CatBoost with Categorical Features
print("\n=== CatBoost with Categorical Features ===")

if CATBOOST_AVAILABLE:
    # Split categorical data
    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )
    
    # Create CatBoost datasets with categorical features
    train_data_cat = cb.Pool(X_train_cat, label=y_train_cat, cat_features=cat_features)
    test_data_cat = cb.Pool(X_test_cat, label=y_test_cat, cat_features=cat_features)
    
    # Set parameters
    params_cat = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'random_seed': 42,
        'verbose': False
    }
    
    # Train model
    print("Training CatBoost with categorical features...")
    model_cat = cb.CatBoostClassifier(**params_cat)
    model_cat.fit(train_data_cat, eval_set=test_data_cat, early_stopping_rounds=10, verbose=False)
    
    # Make predictions
    y_pred_cat = model_cat.predict(X_test_cat)
    y_pred_proba_cat = model_cat.predict_proba(X_test_cat)[:, 1]
    
    # Evaluate
    cat_accuracy = accuracy_score(y_test_cat, y_pred_cat)
    cat_auc = roc_auc_score(y_test_cat, y_pred_proba_cat)
    
    print(f"Categorical Features Accuracy: {cat_accuracy:.4f}")
    print(f"Categorical Features ROC AUC: {cat_auc:.4f}")

# 5. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

if CATBOOST_AVAILABLE:
    # Get feature importance
    importance = model.get_feature_importance()
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
    plt.title('CatBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    for i, (feature, imp) in enumerate(importance_df.head(10).values):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    # Feature importance comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.title('Classification Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(model_reg.feature_importances_)), model_reg.feature_importances_)
    plt.title('Regression Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 6. Hyperparameter Tuning
print("\n=== Hyperparameter Tuning ===")

if CATBOOST_AVAILABLE:
    # Grid Search for hyperparameter tuning
    param_grid = {
        'iterations': [50, 100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    }
    
    # Create CatBoost classifier for sklearn API
    cb_clf = cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=42,
        verbose=False
    )
    
    # Grid search
    print("Performing Grid Search...")
    grid_search = GridSearchCV(
        cb_clf,
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
        cb_clf,
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

# 7. Cross-Validation
print("\n=== Cross-Validation ===")

if CATBOOST_AVAILABLE:
    # Cross-validation with CatBoost
    cb_cv = cb.CatBoostClassifier(
        iterations=100,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        random_seed=42,
        verbose=False
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(cb_cv, X_clf, y_clf, cv=5, scoring='accuracy')
    
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
    plt.title('CatBoost Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 8. Advanced CatBoost Techniques
print("\n=== Advanced CatBoost Techniques ===")

if CATBOOST_AVAILABLE:
    # Ordered boosting
    print("Training with Ordered Boosting...")
    ordered_params = params.copy()
    ordered_params['boosting_type'] = 'Ordered'
    
    ordered_model = cb.CatBoostClassifier(**ordered_params)
    ordered_model.fit(train_data, eval_set=test_data, early_stopping_rounds=10, verbose=False)
    
    ordered_pred = ordered_model.predict(X_test_clf)
    ordered_accuracy = accuracy_score(y_test_clf, ordered_pred)
    print(f"Ordered Boosting Accuracy: {ordered_accuracy:.4f}")
    
    # Plain boosting
    print("\nTraining with Plain Boosting...")
    plain_params = params.copy()
    plain_params['boosting_type'] = 'Plain'
    
    plain_model = cb.CatBoostClassifier(**plain_params)
    plain_model.fit(train_data, eval_set=test_data, early_stopping_rounds=10, verbose=False)
    
    plain_pred = plain_model.predict(X_test_clf)
    plain_accuracy = accuracy_score(y_test_clf, plain_pred)
    print(f"Plain Boosting Accuracy: {plain_accuracy:.4f}")
    
    # Custom loss function (simplified)
    print("\nTraining with custom evaluation metric...")
    custom_params = params.copy()
    
    # Define custom metric
    def custom_metric(approxes, targets, weight):
        """Custom evaluation metric"""
        approx = approxes[0]
        return 'custom_metric', np.mean(np.abs(targets - approx)), False
    
    custom_model = cb.CatBoostClassifier(**custom_params)
    custom_model.fit(train_data, eval_set=test_data, early_stopping_rounds=10, verbose=False)
    
    custom_pred = custom_model.predict(X_test_clf)
    custom_accuracy = accuracy_score(y_test_clf, custom_pred)
    print(f"Custom Metric Model Accuracy: {custom_accuracy:.4f}")

# 9. Model Interpretation and Visualization
print("\n=== Model Interpretation and Visualization ===")

if CATBOOST_AVAILABLE:
    # Learning curves
    evals_result = model.get_evals_result()
    
    plt.figure(figsize=(12, 5))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    if 'validation' in evals_result:
        train_loss = evals_result['learn']['Logloss']
        val_loss = evals_result['validation']['Logloss']
        plt.plot(train_loss, label='Train')
        plt.plot(val_loss, label='Validation')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('CatBoost Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Feature importance comparison
    plt.subplot(1, 2, 2)
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Precision-Recall curve
    y_pred_proba_clf = model.predict_proba(X_test_clf)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test_clf, y_pred_proba_clf)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='CatBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 10. Real-world Dataset Example
print("\n=== Real-world Dataset Example ===")

if CATBOOST_AVAILABLE:
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
        
        # Train CatBoost
        cancer_model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            loss_function='Logloss',
            random_seed=42,
            verbose=False
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

# 11. Model Comparison
print("\n=== Model Comparison ===")

if CATBOOST_AVAILABLE:
    # Compare different CatBoost configurations
    models_comparison = {
        'Default': model,
        'Ordered': ordered_model,
        'Plain': plain_model
    }
    
    comparison_results = {}
    
    for name, model_obj in models_comparison.items():
        pred = model_obj.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, pred)
        comparison_results[name] = acc
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_results.keys(), comparison_results.values(), alpha=0.8)
    plt.title('CatBoost Model Comparison')
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

# 12. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("CatBoost Performance Summary:")
if CATBOOST_AVAILABLE:
    print(f"1. Classification Accuracy: {accuracy:.4f}")
    print(f"2. Regression R² Score: {r2:.4f}")
    print(f"3. Cross-validation Mean Score: {cv_scores.mean():.4f}")
    print(f"4. Categorical Features Accuracy: {cat_accuracy:.4f}")

print(f"\nKey CatBoost Advantages:")
print(f"- Native support for categorical features")
print(f"- Ordered boosting for better generalization")
print(f"- Robust to overfitting")
print(f"- Handles missing values automatically")
print(f"- Fast training and prediction")
print(f"- GPU acceleration support")
print(f"- Built-in feature importance")

print(f"\nBest Practices:")
print(f"1. Start with default parameters and tune gradually")
print(f"2. Use early stopping to prevent overfitting")
print(f"3. Tune learning_rate and iterations together")
print(f"4. Use cross-validation for robust evaluation")
print(f"5. Monitor feature importance for feature selection")
print(f"6. Use Ordered boosting for better generalization")
print(f"7. Handle categorical features properly")
print(f"8. Use appropriate evaluation metrics")
print(f"9. Scale numerical features for better performance")
print(f"10. Use GPU acceleration for large datasets")

print(f"\nCommon Parameters to Tune:")
print(f"- iterations: Number of trees (50-1000)")
print(f"- depth: Tree depth (4-10)")
print(f"- learning_rate: Step size (0.01-0.3)")
print(f"- l2_leaf_reg: L2 regularization (1-10)")
print(f"- border_count: Number of splits (32-255)")
print(f"- boosting_type: 'Ordered' or 'Plain'")

print(f"\nAdvanced Techniques:")
print(f"- Ordered boosting for better generalization")
print(f"- Categorical feature handling")
print(f"- Custom loss functions")
print(f"- Feature importance analysis")
print(f"- GPU acceleration")
print(f"- Missing value handling")
print(f"- Overfitting detection") 