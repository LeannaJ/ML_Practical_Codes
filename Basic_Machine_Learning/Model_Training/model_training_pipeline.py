"""
Model Training Pipeline
======================

This script demonstrates a comprehensive model training pipeline including:
- Data preparation and preprocessing
- Model selection and training
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """Prepare data for training"""
    print(f"=== DATA PREPARATION FOR {target_column.upper()} ===")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 10 else None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline():
    """Create preprocessing pipeline"""
    preprocessing_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    return preprocessing_pipeline

def get_regression_models():
    """Get regression models for comparison"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR()
    }
    
    return models

def get_classification_models():
    """Get classification models for comparison"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVC': SVC(random_state=42)
    }
    
    return models

def train_models(X_train, y_train, models, preprocessing_pipeline=None):
    """Train multiple models and return results"""
    print("=== MODEL TRAINING ===")
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with preprocessing if provided
        if preprocessing_pipeline is not None:
            pipeline = Pipeline([
                ('preprocessing', preprocessing_pipeline),
                ('model', model)
            ])
        else:
            pipeline = model
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2' if 'regression' in str(type(model)).lower() else 'accuracy')
        
        results[name] = {
            'model': pipeline,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }
        
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

def evaluate_regression_models(results, X_test, y_test):
    """Evaluate regression models"""
    print("\n=== REGRESSION MODEL EVALUATION ===")
    
    evaluation_results = {}
    
    for name, result in results.items():
        model = result['model']
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        evaluation_results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"\n{name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
    
    return evaluation_results

def evaluate_classification_models(results, X_test, y_test):
    """Evaluate classification models"""
    print("\n=== CLASSIFICATION MODEL EVALUATION ===")
    
    evaluation_results = {}
    
    for name, result in results.items():
        model = result['model']
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        evaluation_results[name] = {
            'Accuracy': accuracy,
            'Predictions': y_pred
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("  Confusion Matrix:")
        print(cm)
    
    return evaluation_results

def hyperparameter_tuning(X_train, y_train, model, param_grid, cv=5, n_iter=20):
    """Perform hyperparameter tuning"""
    print(f"\n=== HYPERPARAMETER TUNING ===")
    
    # Grid search for small parameter spaces
    if len(param_grid) <= 10:
        print("Using GridSearchCV...")
        search = GridSearchCV(
            model, param_grid, cv=cv, scoring='r2' if 'regression' in str(type(model)).lower() else 'accuracy',
            n_jobs=-1, verbose=1
        )
    else:
        print("Using RandomizedSearchCV...")
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv, 
            scoring='r2' if 'regression' in str(type(model)).lower() else 'accuracy',
            n_jobs=-1, verbose=1, random_state=42
        )
    
    search.fit(X_train, y_train)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best cross-validation score: {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_params_, search.best_score_

def plot_model_comparison(evaluation_results, metric='R2'):
    """Plot model comparison"""
    print(f"\n=== MODEL COMPARISON ({metric}) ===")
    
    models = list(evaluation_results.keys())
    scores = [evaluation_results[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, scores)
    plt.title(f'Model Comparison - {metric}')
    plt.xlabel('Models')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.4f}', ha='center', va='bottom')
    
    # Horizontal bar plot
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(models))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, models)
    plt.xlabel(metric)
    plt.title(f'Model Comparison - {metric} (Horizontal)')
    
    plt.tight_layout()
    plt.show()
    
    # Print ranking
    print(f"\nModel Ranking by {metric}:")
    ranking = sorted(evaluation_results.items(), key=lambda x: x[1][metric], reverse=True)
    for i, (model, metrics) in enumerate(ranking, 1):
        print(f"{i}. {model}: {metrics[metric]:.4f}")

def save_model(model, filename):
    """Save trained model"""
    print(f"\n=== SAVING MODEL ===")
    
    try:
        joblib.dump(model, filename)
        print(f"Model saved successfully as {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename):
    """Load trained model"""
    print(f"\n=== LOADING MODEL ===")
    
    try:
        model = joblib.load(filename)
        print(f"Model loaded successfully from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def regression_pipeline():
    """Complete regression pipeline"""
    print("=== REGRESSION PIPELINE ===\n")
    
    # 1. Create dataset
    print("1. Creating regression dataset...")
    df_regression, _ = create_sample_datasets()
    
    # 2. Prepare data
    print("\n2. Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df_regression, 'purchase_amount')
    
    # 3. Create preprocessing pipeline
    print("\n3. Creating preprocessing pipeline...")
    preprocessing_pipeline = create_preprocessing_pipeline()
    
    # 4. Get models
    print("\n4. Getting regression models...")
    models = get_regression_models()
    
    # 5. Train models
    print("\n5. Training models...")
    results = train_models(X_train, y_train, models, preprocessing_pipeline)
    
    # 6. Evaluate models
    print("\n6. Evaluating models...")
    evaluation_results = evaluate_regression_models(results, X_test, y_test)
    
    # 7. Plot comparison
    plot_model_comparison(evaluation_results, 'R2')
    
    # 8. Hyperparameter tuning for best model
    print("\n7. Hyperparameter tuning for best model...")
    best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['R2'])[0]
    best_model = results[best_model_name]['model']
    
    # Example parameter grid for Random Forest
    if 'Random Forest' in best_model_name:
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
        tuned_model, best_params, best_score = hyperparameter_tuning(
            X_train, y_train, best_model, param_grid
        )
        
        # Save best model
        save_model(tuned_model, 'best_regression_model.pkl')
    
    return results, evaluation_results

def classification_pipeline():
    """Complete classification pipeline"""
    print("\n=== CLASSIFICATION PIPELINE ===\n")
    
    # 1. Create dataset
    print("1. Creating classification dataset...")
    _, df_classification = create_sample_datasets()
    
    # 2. Prepare data
    print("\n2. Preparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df_classification, 'high_value_customer')
    
    # 3. Create preprocessing pipeline
    print("\n3. Creating preprocessing pipeline...")
    preprocessing_pipeline = create_preprocessing_pipeline()
    
    # 4. Get models
    print("\n4. Getting classification models...")
    models = get_classification_models()
    
    # 5. Train models
    print("\n5. Training models...")
    results = train_models(X_train, y_train, models, preprocessing_pipeline)
    
    # 6. Evaluate models
    print("\n6. Evaluating models...")
    evaluation_results = evaluate_classification_models(results, X_test, y_test)
    
    # 7. Plot comparison
    plot_model_comparison(evaluation_results, 'Accuracy')
    
    # 8. Hyperparameter tuning for best model
    print("\n7. Hyperparameter tuning for best model...")
    best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['Accuracy'])[0]
    best_model = results[best_model_name]['model']
    
    # Example parameter grid for Random Forest
    if 'Random Forest' in best_model_name:
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5, 10]
        }
        tuned_model, best_params, best_score = hyperparameter_tuning(
            X_train, y_train, best_model, param_grid
        )
        
        # Save best model
        save_model(tuned_model, 'best_classification_model.pkl')
    
    return results, evaluation_results

def main():
    """Main function to run complete model training pipelines"""
    print("=== COMPREHENSIVE MODEL TRAINING PIPELINE ===\n")
    
    # Run regression pipeline
    regression_results, regression_evaluation = regression_pipeline()
    
    # Run classification pipeline
    classification_results, classification_evaluation = classification_pipeline()
    
    print("\n" + "="*60)
    print("=== MODEL TRAINING COMPLETE! ===")
    print("Both regression and classification pipelines have been completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main() 