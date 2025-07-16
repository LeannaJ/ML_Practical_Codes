"""
ML Pipeline Examples
===================

- Complete ML Pipeline with scikit-learn
- MLflow Pipeline Management
- Custom Pipeline Components
- Pipeline Serialization and Loading
- Pipeline Versioning
- Automated Pipeline Execution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# MLflow for experiment tracking
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Install with: pip install mlflow")

print("=== ML Pipeline Examples ===")

# 1. Basic ML Pipeline
print("\n=== Basic ML Pipeline ===")

def create_basic_pipeline():
    """Create a basic ML pipeline"""
    
    # Define preprocessing steps
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selector', SelectKBest(score_func=f_classif, k=10))
    ])
    
    # Define the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                         n_redundant=3, n_repeated=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Create and train pipeline
print("Creating and training basic pipeline...")
basic_pipeline = create_basic_pipeline()
basic_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = basic_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Basic Pipeline Accuracy: {accuracy:.4f}")

# 2. Advanced Pipeline with Custom Components
print("\n=== Advanced Pipeline with Custom Components ===")

from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom feature engineering transformer"""
    
    def __init__(self, add_polynomial=False, add_interactions=False):
        self.add_polynomial = add_polynomial
        self.add_interactions = add_interactions
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        if self.add_polynomial:
            # Add polynomial features
            for i in range(X.shape[1]):
                X_transformed = np.column_stack([X_transformed, X[:, i]**2])
        
        if self.add_interactions:
            # Add interaction features
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    X_transformed = np.column_stack([X_transformed, X[:, i] * X[:, j]])
        
        return X_transformed

class ModelEvaluator(BaseEstimator, TransformerMixin):
    """Custom model evaluator"""
    
    def __init__(self, metrics=['accuracy', 'precision', 'recall']):
        self.metrics = metrics
        self.results = {}
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # This transformer doesn't modify the data
        return X
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        self.results = {}
        
        if 'accuracy' in self.metrics:
            self.results['accuracy'] = accuracy_score(y_true, y_pred)
        if 'precision' in self.metrics:
            self.results['precision'] = precision_score(y_true, y_pred, average='weighted')
        if 'recall' in self.metrics:
            self.results['recall'] = recall_score(y_true, y_pred, average='weighted')
            
        return self.results

# Create advanced pipeline
def create_advanced_pipeline():
    """Create an advanced ML pipeline with custom components"""
    
    # Define preprocessing steps
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('feature_engineer', FeatureEngineer(add_polynomial=True, add_interactions=True)),
        ('scaler', StandardScaler()),
        ('feature_selector', SelectKBest(score_func=f_classif, k=15))
    ])
    
    # Define the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('evaluator', ModelEvaluator())
    ])
    
    return pipeline

# Train advanced pipeline
print("Creating and training advanced pipeline...")
advanced_pipeline = create_advanced_pipeline()
advanced_pipeline.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_advanced = advanced_pipeline.predict(X_test)
evaluator = advanced_pipeline.named_steps['evaluator']
results = evaluator.evaluate(y_test, y_pred_advanced)

print("Advanced Pipeline Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# 3. Pipeline Serialization and Loading
print("\n=== Pipeline Serialization and Loading ===")

# Save pipeline using joblib
pipeline_path = "basic_pipeline.joblib"
print(f"Saving pipeline to {pipeline_path}...")
joblib.dump(basic_pipeline, pipeline_path)

# Load pipeline
print(f"Loading pipeline from {pipeline_path}...")
loaded_pipeline = joblib.load(pipeline_path)

# Test loaded pipeline
y_pred_loaded = loaded_pipeline.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"Loaded Pipeline Accuracy: {accuracy_loaded:.4f}")

# Save pipeline using pickle
pipeline_path_pkl = "basic_pipeline.pkl"
print(f"Saving pipeline to {pipeline_path_pkl}...")
with open(pipeline_path_pkl, 'wb') as f:
    pickle.dump(basic_pipeline, f)

# Load pipeline using pickle
print(f"Loading pipeline from {pipeline_path_pkl}...")
with open(pipeline_path_pkl, 'rb') as f:
    loaded_pipeline_pkl = pickle.load(f)

# Test loaded pipeline
y_pred_loaded_pkl = loaded_pipeline_pkl.predict(X_test)
accuracy_loaded_pkl = accuracy_score(y_test, y_pred_loaded_pkl)
print(f"Pickle Loaded Pipeline Accuracy: {accuracy_loaded_pkl:.4f}")

# 4. Pipeline Configuration Management
print("\n=== Pipeline Configuration Management ===")

class PipelineConfig:
    """Pipeline configuration manager"""
    
    def __init__(self, config_path=None):
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.set_default_config()
    
    def set_default_config(self):
        """Set default pipeline configuration"""
        self.config = {
            'data': {
                'test_size': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'imputation_strategy': 'mean',
                'scaling': True,
                'feature_selection': True,
                'n_features': 10
            },
            'model': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'max_depth': None,
                'random_state': 42
            },
            'evaluation': {
                'cv_folds': 5,
                'metrics': ['accuracy', 'precision', 'recall', 'f1']
            }
        }
    
    def load_config(self, config_path):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def save_config(self, config_path):
        """Save configuration to file"""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get_config(self):
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates):
        """Update configuration"""
        self.config.update(updates)

# Create and save configuration
config = PipelineConfig()
config_path = "pipeline_config.json"
config.save_config(config_path)
print(f"Configuration saved to {config_path}")

# Load configuration
loaded_config = PipelineConfig(config_path)
print("Loaded configuration:")
print(json.dumps(loaded_config.get_config(), indent=2))

# 5. MLflow Pipeline Integration
print("\n=== MLflow Pipeline Integration ===")

if MLFLOW_AVAILABLE:
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Start MLflow run
    with mlflow.start_run(run_name="ml_pipeline_experiment"):
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("test_size", 0.2)
        
        # Train pipeline
        pipeline_mlflow = create_basic_pipeline()
        pipeline_mlflow.fit(X_train, y_train)
        
        # Make predictions
        y_pred_mlflow = pipeline_mlflow.predict(X_test)
        accuracy_mlflow = accuracy_score(y_test, y_pred_mlflow)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_mlflow)
        
        # Log model
        mlflow.sklearn.log_model(pipeline_mlflow, "pipeline_model")
        
        # Log artifacts
        with open("pipeline_results.txt", "w") as f:
            f.write(f"Accuracy: {accuracy_mlflow:.4f}\n")
            f.write(classification_report(y_test, y_pred_mlflow))
        
        mlflow.log_artifact("pipeline_results.txt")
        
        print(f"MLflow Pipeline Accuracy: {accuracy_mlflow:.4f}")
        print("Model and artifacts logged to MLflow")

# 6. Pipeline Versioning
print("\n=== Pipeline Versioning ===")

class PipelineVersionManager:
    """Pipeline version manager"""
    
    def __init__(self, base_path="pipeline_versions"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        self.version_file = os.path.join(base_path, "versions.json")
        self.load_versions()
    
    def load_versions(self):
        """Load version information"""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def save_versions(self):
        """Save version information"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=4)
    
    def save_pipeline_version(self, pipeline, version, description=""):
        """Save a new pipeline version"""
        version_path = os.path.join(self.base_path, f"pipeline_v{version}")
        os.makedirs(version_path, exist_ok=True)
        
        # Save pipeline
        pipeline_file = os.path.join(version_path, "pipeline.joblib")
        joblib.dump(pipeline, pipeline_file)
        
        # Save metadata
        metadata = {
            'version': version,
            'description': description,
            'created_at': pd.Timestamp.now().isoformat(),
            'pipeline_file': pipeline_file
        }
        
        metadata_file = os.path.join(version_path, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Update versions
        self.versions[version] = metadata
        self.save_versions()
        
        print(f"Pipeline version {version} saved successfully")
    
    def load_pipeline_version(self, version):
        """Load a specific pipeline version"""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")
        
        pipeline_file = self.versions[version]['pipeline_file']
        return joblib.load(pipeline_file)
    
    def list_versions(self):
        """List all available versions"""
        return list(self.versions.keys())
    
    def get_version_info(self, version):
        """Get information about a specific version"""
        return self.versions.get(version, None)

# Create version manager and save pipeline versions
version_manager = PipelineVersionManager()

# Save different pipeline versions
version_manager.save_pipeline_version(
    basic_pipeline, 
    "1.0", 
    "Basic pipeline with Random Forest"
)

version_manager.save_pipeline_version(
    advanced_pipeline, 
    "2.0", 
    "Advanced pipeline with feature engineering"
)

# List versions
print("Available pipeline versions:")
for version in version_manager.list_versions():
    info = version_manager.get_version_info(version)
    print(f"Version {version}: {info['description']}")

# Load and test a specific version
loaded_v1 = version_manager.load_pipeline_version("1.0")
y_pred_v1 = loaded_v1.predict(X_test)
accuracy_v1 = accuracy_score(y_test, y_pred_v1)
print(f"Version 1.0 Accuracy: {accuracy_v1:.4f}")

# 7. Automated Pipeline Execution
print("\n=== Automated Pipeline Execution ===")

class AutomatedPipeline:
    """Automated pipeline execution"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def run_pipeline(self, X, y):
        """Run complete pipeline"""
        print("Starting automated pipeline execution...")
        
        # Data splitting
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create pipeline based on configuration
        pipeline = self._create_pipeline_from_config()
        
        # Train pipeline
        print("Training pipeline...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        print("Evaluating model...")
        self.results = self._evaluate_model(y_test, y_pred)
        
        # Cross-validation
        if self.config['evaluation']['cv_folds'] > 1:
            print("Performing cross-validation...")
            cv_scores = cross_val_score(
                pipeline, X, y, 
                cv=self.config['evaluation']['cv_folds'],
                scoring='accuracy'
            )
            self.results['cv_mean'] = cv_scores.mean()
            self.results['cv_std'] = cv_scores.std()
        
        return pipeline, self.results
    
    def _create_pipeline_from_config(self):
        """Create pipeline from configuration"""
        preprocessing_steps = []
        
        # Add imputation
        if 'imputation_strategy' in self.config['preprocessing']:
            preprocessing_steps.append((
                'imputer', 
                SimpleImputer(strategy=self.config['preprocessing']['imputation_strategy'])
            ))
        
        # Add scaling
        if self.config['preprocessing']['scaling']:
            preprocessing_steps.append(('scaler', StandardScaler()))
        
        # Add feature selection
        if self.config['preprocessing']['feature_selection']:
            preprocessing_steps.append((
                'feature_selector', 
                SelectKBest(score_func=f_classif, k=self.config['preprocessing']['n_features'])
            ))
        
        # Create preprocessing pipeline
        preprocessor = Pipeline(preprocessing_steps)
        
        # Create model
        if self.config['model']['algorithm'] == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=self.config['model']['n_estimators'],
                max_depth=self.config['model']['max_depth'],
                random_state=self.config['model']['random_state']
            )
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        return pipeline
    
    def _evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results = {}
        metrics = self.config['evaluation']['metrics']
        
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        if 'precision' in metrics:
            results['precision'] = precision_score(y_true, y_pred, average='weighted')
        if 'recall' in metrics:
            results['recall'] = recall_score(y_true, y_pred, average='weighted')
        if 'f1' in metrics:
            results['f1'] = f1_score(y_true, y_pred, average='weighted')
        
        return results
    
    def save_results(self, filepath):
        """Save pipeline results"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {filepath}")

# Run automated pipeline
automated_pipeline = AutomatedPipeline(loaded_config.get_config())
final_pipeline, final_results = automated_pipeline.run_pipeline(X, y)

print("\nAutomated Pipeline Results:")
for metric, value in final_results.items():
    print(f"{metric}: {value:.4f}")

# Save results
automated_pipeline.save_results("automated_pipeline_results.json")

# 8. Pipeline Visualization
print("\n=== Pipeline Visualization ===")

# Visualize pipeline structure
plt.figure(figsize=(12, 8))

# Create a simple visualization of the pipeline
pipeline_steps = ['Data Input', 'Imputation', 'Scaling', 'Feature Selection', 'Model Training', 'Prediction']
step_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']

plt.subplot(2, 2, 1)
for i, (step, color) in enumerate(zip(pipeline_steps, step_colors)):
    plt.barh(i, 1, color=color, alpha=0.7)
    plt.text(0.5, i, step, ha='center', va='center', fontweight='bold')

plt.title('Pipeline Structure')
plt.xlabel('Steps')
plt.yticks(range(len(pipeline_steps)), pipeline_steps)
plt.xlim(0, 1)

# Performance comparison
plt.subplot(2, 2, 2)
pipelines = ['Basic', 'Advanced', 'Automated']
accuracies = [accuracy, results['accuracy'], final_results['accuracy']]

bars = plt.bar(pipelines, accuracies, alpha=0.7)
plt.title('Pipeline Performance Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{acc:.3f}', ha='center', va='bottom')

# Feature importance (if available)
plt.subplot(2, 2, 3)
if hasattr(basic_pipeline.named_steps['classifier'], 'feature_importances_'):
    feature_importance = basic_pipeline.named_steps['classifier'].feature_importances_
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')

# Pipeline execution time (simulated)
plt.subplot(2, 2, 4)
execution_times = [0.5, 1.2, 0.8]  # Simulated times
bars = plt.bar(pipelines, execution_times, alpha=0.7)
plt.title('Pipeline Execution Time')
plt.ylabel('Time (seconds)')

for bar, time_val in zip(bars, execution_times):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{time_val}s', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 9. Summary and Best Practices
print("\n=== Summary and Best Practices ===")

print("Pipeline Performance Summary:")
print("=" * 50)
print(f"Basic Pipeline Accuracy: {accuracy:.4f}")
print(f"Advanced Pipeline Accuracy: {results['accuracy']:.4f}")
print(f"Automated Pipeline Accuracy: {final_results['accuracy']:.4f}")

print("\nKey Pipeline Components:")
print("=" * 50)
print("1. Data Preprocessing: Imputation, Scaling, Feature Selection")
print("2. Model Training: Algorithm selection and hyperparameter tuning")
print("3. Model Evaluation: Cross-validation and multiple metrics")
print("4. Model Serialization: Saving and loading trained models")
print("5. Configuration Management: Parameter management and versioning")
print("6. Experiment Tracking: MLflow integration for reproducibility")
print("7. Automated Execution: End-to-end pipeline automation")

print("\nBest Practices:")
print("=" * 50)
print("1. Always use pipelines to ensure consistent preprocessing")
print("2. Implement proper train/test splits and cross-validation")
print("3. Use configuration files for parameter management")
print("4. Version your pipelines and models")
print("5. Track experiments with MLflow or similar tools")
print("6. Implement automated testing for pipeline components")
print("7. Use proper serialization methods (joblib for scikit-learn)")
print("8. Monitor pipeline performance and execution time")
print("9. Implement error handling and logging")
print("10. Document pipeline components and dependencies")

print("\nPipeline Development Workflow:")
print("=" * 50)
print("1. Define pipeline requirements and objectives")
print("2. Create modular pipeline components")
print("3. Implement configuration management")
print("4. Add experiment tracking and logging")
print("5. Test pipeline with different datasets")
print("6. Optimize performance and resource usage")
print("7. Deploy pipeline to production environment")
print("8. Monitor and maintain pipeline performance") 