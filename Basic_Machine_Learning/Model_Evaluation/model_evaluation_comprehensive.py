"""
Comprehensive Model Evaluation
=============================

This script demonstrates comprehensive model evaluation techniques including:
- Regression metrics (MSE, RMSE, MAE, R², Adjusted R²)
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Cross-validation strategies
- Learning curves
- Residual analysis
- Feature importance analysis
- Model interpretability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_datasets():
    """Create sample datasets for regression and classification"""
    n_samples = 1000
    
    # Generate features
    np.random.seed(42)
    age = np.random.normal(35, 10, n_samples)
    income = 50000 + age * 1000 + np.random.normal(0, 5000, n_samples)
    education_years = np.random.choice([12, 16, 18, 22], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    credit_score = np.random.normal(700, 100, n_samples)
    
    # Create regression dataset
    regression_data = {
        'age': age,
        'income': income,
        'education_years': education_years,
        'credit_score': credit_score,
        'purchase_amount': 100 + 0.5 * age + 0.001 * income + 5 * education_years + 0.1 * credit_score + np.random.normal(0, 10, n_samples)
    }
    
    # Create classification dataset
    classification_data = {
        'age': age,
        'income': income,
        'education_years': education_years,
        'credit_score': credit_score,
        'high_value_customer': ((income > income.mean()) & (credit_score > credit_score.mean())).astype(int)
    }
    
    df_regression = pd.DataFrame(regression_data)
    df_classification = pd.DataFrame(classification_data)
    
    return df_regression, df_classification

def regression_metrics(y_true, y_pred, X_test=None):
    """Calculate comprehensive regression metrics"""
    print("=== REGRESSION METRICS ===\n")
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R²
    if X_test is not None:
        n = len(y_true)
        p = X_test.shape[1]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adjusted_r2 = None
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Print results
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    if adjusted_r2 is not None:
        print(f"Adjusted R² Score: {adjusted_r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Adjusted_R2': adjusted_r2,
        'MAPE': mape
    }

def classification_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive classification metrics"""
    print("=== CLASSIFICATION METRICS ===\n")
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC (if probabilities are available)
    roc_auc = None
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) == 1:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        else:
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'ROC_AUC': roc_auc,
        'Confusion_Matrix': cm
    }

def cross_validation_evaluation(X, y, model, cv=5, scoring='r2'):
    """Perform cross-validation evaluation"""
    print(f"=== CROSS-VALIDATION EVALUATION ({cv}-fold) ===\n")
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Calculate statistics
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {mean_score:.4f} (+/- {std_score * 2:.4f})")
    print(f"CV score range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
    
    return {
        'cv_scores': cv_scores,
        'mean_score': mean_score,
        'std_score': std_score
    }

def plot_learning_curves(X, y, model, title="Learning Curves"):
    """Plot learning curves to analyze bias/variance"""
    print(f"=== LEARNING CURVES ===\n")
    
    # Calculate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2' if 'regression' in str(type(model)).lower() else 'accuracy'
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    # Analyze bias/variance
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    gap = final_train_score - final_val_score
    
    print(f"Final training score: {final_train_score:.4f}")
    print(f"Final validation score: {final_val_score:.4f}")
    print(f"Gap (Training - Validation): {gap:.4f}")
    
    if gap > 0.1:
        print("High variance detected - model is overfitting")
    elif final_val_score < 0.6:
        print("High bias detected - model is underfitting")
    else:
        print("Good balance between bias and variance")

def plot_validation_curves(X, y, model, param_name, param_range, title="Validation Curves"):
    """Plot validation curves for hyperparameter tuning"""
    print(f"=== VALIDATION CURVES ===\n")
    
    # Calculate validation curves
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring='r2' if 'regression' in str(type(model)).lower() else 'accuracy'
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot validation curves
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(param_range, val_mean, label='Cross-validation score', color='red', marker='s')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    # Find optimal parameter
    optimal_idx = np.argmax(val_mean)
    optimal_param = param_range[optimal_idx]
    optimal_score = val_mean[optimal_idx]
    
    print(f"Optimal {param_name}: {optimal_param}")
    print(f"Optimal validation score: {optimal_score:.4f}")

def residual_analysis(y_true, y_pred, X_test=None):
    """Perform residual analysis for regression models"""
    print("=== RESIDUAL ANALYSIS ===\n")
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Basic statistics
    print(f"Residual Statistics:")
    print(f"Mean: {np.mean(residuals):.4f}")
    print(f"Std: {np.std(residuals):.4f}")
    print(f"Min: {np.min(residuals):.4f}")
    print(f"Max: {np.max(residuals):.4f}")
    
    # Create residual plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted Values')
    axes[0, 0].grid(True)
    
    # 2. Residuals histogram
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True)
    
    # 3. Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    axes[1, 0].grid(True)
    
    # 4. Residuals vs Index
    axes[1, 1].plot(residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Sample Index')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Normality test
    from scipy.stats import shapiro
    stat, p_value = shapiro(residuals)
    print(f"\nShapiro-Wilk normality test:")
    print(f"Statistic: {stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Residuals are {'normal' if p_value > 0.05 else 'not normal'} (α=0.05)")

def feature_importance_analysis(model, X, y, feature_names=None):
    """Analyze feature importance"""
    print("=== FEATURE IMPORTANCE ANALYSIS ===\n")
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        # Tree-based models
        importances = model.feature_importances_
        method = "Built-in feature importance"
    elif hasattr(model, 'coef_'):
        # Linear models
        importances = np.abs(model.coef_)
        if len(importances.shape) > 1:
            importances = np.mean(importances, axis=0)
        method = "Coefficient magnitude"
    else:
        # Permutation importance
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        importances = result.importances_mean
        method = "Permutation importance"
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(f"Feature importance method: {method}")
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """Plot ROC curve for classification"""
    print("=== ROC CURVE ===\n")
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    print(f"AUC Score: {auc_score:.4f}")

def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """Plot precision-recall curve for classification"""
    print("=== PRECISION-RECALL CURVE ===\n")
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True)
    plt.show()

def comprehensive_regression_evaluation(X, y, model):
    """Complete regression model evaluation"""
    print("=== COMPREHENSIVE REGRESSION EVALUATION ===\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 1. Basic metrics
    metrics = regression_metrics(y_test, y_pred, X_test)
    
    # 2. Cross-validation
    cv_results = cross_validation_evaluation(X, y, model)
    
    # 3. Learning curves
    plot_learning_curves(X, y, model, "Regression Learning Curves")
    
    # 4. Residual analysis
    residual_analysis(y_test, y_pred, X_test)
    
    # 5. Feature importance
    importance_df = feature_importance_analysis(model, X, y, X.columns.tolist())
    
    return {
        'metrics': metrics,
        'cv_results': cv_results,
        'importance': importance_df
    }

def comprehensive_classification_evaluation(X, y, model):
    """Complete classification model evaluation"""
    print("=== COMPREHENSIVE CLASSIFICATION EVALUATION ===\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 1. Basic metrics
    metrics = classification_metrics(y_test, y_pred, y_pred_proba)
    
    # 2. Cross-validation
    cv_results = cross_validation_evaluation(X, y, model, scoring='accuracy')
    
    # 3. Learning curves
    plot_learning_curves(X, y, model, "Classification Learning Curves")
    
    # 4. ROC curve
    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba)
        plot_precision_recall_curve(y_test, y_pred_proba)
    
    # 5. Feature importance
    importance_df = feature_importance_analysis(model, X, y, X.columns.tolist())
    
    return {
        'metrics': metrics,
        'cv_results': cv_results,
        'importance': importance_df
    }

def main():
    """Main function to run comprehensive model evaluation"""
    print("=== COMPREHENSIVE MODEL EVALUATION DEMONSTRATION ===\n")
    
    # Create datasets
    print("1. Creating sample datasets...")
    df_regression, df_classification = create_sample_datasets()
    
    # Regression evaluation
    print("\n" + "="*60)
    print("2. Regression Model Evaluation")
    print("="*60)
    
    X_reg = df_regression.drop('purchase_amount', axis=1)
    y_reg = df_regression['purchase_amount']
    
    regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
    regression_results = comprehensive_regression_evaluation(X_reg, y_reg, regression_model)
    
    # Classification evaluation
    print("\n" + "="*60)
    print("3. Classification Model Evaluation")
    print("="*60)
    
    X_clf = df_classification.drop('high_value_customer', axis=1)
    y_clf = df_classification['high_value_customer']
    
    classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
    classification_results = comprehensive_classification_evaluation(X_clf, y_clf, classification_model)
    
    print("\n" + "="*60)
    print("=== MODEL EVALUATION COMPLETE! ===")
    print("Comprehensive evaluation of both regression and classification models completed!")
    print("="*60)

if __name__ == "__main__":
    main() 