"""
Machine Learning Utilities
=========================

This script contains commonly used utility functions for machine learning including:
- Data loading and saving utilities
- Visualization helpers
- Performance monitoring
- Model comparison utilities
- Data validation functions
- Configuration management
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MLUtilities:
    """Collection of machine learning utility functions"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ==================== DATA UTILITIES ====================
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from various file formats"""
        print(f"Loading data from: {file_path}")
        
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_extension == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif file_extension == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_extension == 'pickle' or file_extension == 'pkl':
                df = pd.read_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, file_path: str, **kwargs) -> bool:
        """Save data to various file formats"""
        print(f"Saving data to: {file_path}")
        
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_extension in ['xlsx', 'xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_extension == 'json':
                df.to_json(file_path, orient='records', **kwargs)
            elif file_extension == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            elif file_extension == 'pickle' or file_extension == 'pkl':
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Data saved successfully to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def save_model(self, model: Any, file_path: str) -> bool:
        """Save trained model"""
        print(f"Saving model to: {file_path}")
        
        try:
            joblib.dump(model, file_path)
            print(f"Model saved successfully to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, file_path: str) -> Any:
        """Load trained model"""
        print(f"Loading model from: {file_path}")
        
        try:
            model = joblib.load(file_path)
            print(f"Model loaded successfully from {file_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    # ==================== VISUALIZATION UTILITIES ====================
    
    def plot_correlation_matrix(self, df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Plot correlation matrix heatmap"""
        plt.figure(figsize=figsize)
        
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, df: pd.DataFrame, columns: List[str] = None, 
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """Plot distributions of numerical features"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, float], metric: str = 'Score', 
                            figsize: Tuple[int, int] = (12, 6)) -> None:
        """Plot model comparison bar chart"""
        plt.figure(figsize=figsize)
        
        models = list(results.keys())
        scores = list(results.values())
        
        bars = plt.bar(models, scores)
        plt.title(f'Model Comparison - {metric}')
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, train_sizes: np.ndarray, train_scores: np.ndarray, 
                           val_scores: np.ndarray, title: str = "Learning Curves") -> None:
        """Plot learning curves"""
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
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
    
    # ==================== PERFORMANCE MONITORING ====================
    
    def timing_decorator(self, func):
        """Decorator to measure function execution time"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        return wrapper
    
    def memory_usage(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate memory usage of DataFrame"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            'total_memory_mb': total_memory / 1024**2,
            'memory_per_column': memory_usage.to_dict()
        }
    
    def print_memory_usage(self, df: pd.DataFrame) -> None:
        """Print memory usage information"""
        memory_info = self.memory_usage(df)
        print(f"Total memory usage: {memory_info['total_memory_mb']:.2f} MB")
        
        print("\nMemory usage per column:")
        for col, memory in memory_info['memory_per_column'].items():
            print(f"  {col}: {memory / 1024**2:.2f} MB")
    
    # ==================== DATA VALIDATION ====================
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate DataFrame and return summary statistics"""
        validation_results = {
            'shape': df.shape,
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_per_column': {col: df[col].nunique() for col in df.columns},
            'memory_usage_mb': self.memory_usage(df)['total_memory_mb']
        }
        
        return validation_results
    
    def print_validation_summary(self, df: pd.DataFrame) -> None:
        """Print comprehensive data validation summary"""
        print("=== DATA VALIDATION SUMMARY ===\n")
        
        validation = self.validate_dataframe(df)
        
        print(f"Dataset Shape: {validation['shape']}")
        print(f"Memory Usage: {validation['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {validation['duplicate_rows']}")
        
        print("\nData Types:")
        for dtype, count in validation['data_types'].items():
            print(f"  {dtype}: {count}")
        
        print("\nMissing Values:")
        for col, missing in validation['missing_values'].items():
            if missing > 0:
                percentage = validation['missing_percentage'][col]
                print(f"  {col}: {missing} ({percentage:.2f}%)")
        
        print("\nUnique Values per Column:")
        for col, unique_count in validation['unique_values_per_column'].items():
            print(f"  {col}: {unique_count}")
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check data quality issues"""
        issues = {
            'missing_data': [],
            'duplicate_data': [],
            'outliers': [],
            'inconsistent_data': []
        }
        
        # Check missing data
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            if missing_pct > 20:
                issues['missing_data'].append(f"{col}: {missing_pct:.1f}% missing")
        
        # Check duplicate data
        if df.duplicated().sum() > 0:
            issues['duplicate_data'].append(f"{df.duplicated().sum()} duplicate rows found")
        
        # Check outliers in numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                issues['outliers'].append(f"{col}: {len(outliers)} outliers detected")
        
        return issues
    
    # ==================== MODEL UTILITIES ====================
    
    def compare_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5, scoring: str = 'r2') -> Dict[str, float]:
        """Compare multiple models using cross-validation"""
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            # Perform cross-validation
            cv_scores = np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring))
            results[name] = cv_scores
            
            print(f"  {scoring.upper()}: {cv_scores:.4f}")
        
        return results
    
    def get_best_model(self, results: Dict[str, float]) -> Tuple[str, float]:
        """Get the best performing model from results"""
        best_model = max(results.items(), key=lambda x: x[1])
        return best_model
    
    def save_experiment_results(self, results: Dict[str, Any], filename: str = None) -> None:
        """Save experiment results to JSON file"""
        if filename is None:
            filename = f"experiment_results_{self.timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Experiment results saved to {filename}")
    
    # ==================== CONFIGURATION MANAGEMENT ====================
    
    def save_config(self, config: Dict[str, Any], filename: str = None) -> None:
        """Save configuration to JSON file"""
        if filename is None:
            filename = f"config_{self.timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to {filename}")
    
    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        print(f"Configuration loaded from {filename}")
        return config
    
    # ==================== HELPER FUNCTIONS ====================
    
    def create_sample_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Create a sample dataset for testing"""
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'education_years': np.random.choice([12, 16, 18, 22], n_samples),
            'satisfaction_score': np.random.randint(1, 11, n_samples),
            'purchase_amount': np.random.normal(100, 30, n_samples),
            'city': np.random.choice(['New York', 'Los Angeles', 'Chicago'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add some missing values
        df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
        
        return df
    
    def print_separator(self, title: str = "", length: int = 60) -> None:
        """Print a formatted separator"""
        if title:
            print(f"\n{'='*length}")
            print(f"{title:^{length}}")
            print(f"{'='*length}")
        else:
            print(f"\n{'='*length}")

# Example usage and demonstration
def demonstrate_utilities():
    """Demonstrate the utility functions"""
    print("=== MACHINE LEARNING UTILITIES DEMONSTRATION ===\n")
    
    # Initialize utilities
    utils = MLUtilities()
    
    # Create sample dataset
    print("1. Creating sample dataset...")
    df = utils.create_sample_dataset(1000)
    print(f"Dataset created with shape: {df.shape}")
    
    # Data validation
    utils.print_separator("DATA VALIDATION")
    utils.print_validation_summary(df)
    
    # Check data quality
    utils.print_separator("DATA QUALITY CHECK")
    quality_issues = utils.check_data_quality(df)
    for issue_type, issues in quality_issues.items():
        if issues:
            print(f"\n{issue_type.upper()}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n{issue_type.upper()}: No issues found")
    
    # Memory usage
    utils.print_separator("MEMORY USAGE")
    utils.print_memory_usage(df)
    
    # Visualization
    utils.print_separator("VISUALIZATION")
    print("Plotting feature distributions...")
    utils.plot_feature_distributions(df, ['age', 'income', 'satisfaction_score'])
    
    print("Plotting correlation matrix...")
    utils.plot_correlation_matrix(df[['age', 'income', 'education_years', 'satisfaction_score', 'purchase_amount']])
    
    # Model comparison example
    utils.print_separator("MODEL COMPARISON")
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    X = df[['age', 'income', 'education_years']].fillna(df[['age', 'income', 'education_years']].mean())
    y = df['purchase_amount']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = utils.compare_models(models, X, y)
    utils.plot_model_comparison(results, 'R² Score')
    
    # Save results
    utils.print_separator("SAVING RESULTS")
    experiment_results = {
        'dataset_shape': df.shape,
        'model_results': results,
        'quality_issues': quality_issues,
        'timestamp': utils.timestamp
    }
    utils.save_experiment_results(experiment_results)
    
    print("\n=== UTILITIES DEMONSTRATION COMPLETE! ===")

if __name__ == "__main__":
    demonstrate_utilities() 