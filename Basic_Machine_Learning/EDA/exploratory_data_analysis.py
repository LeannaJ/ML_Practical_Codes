"""
Exploratory Data Analysis (EDA)
==============================

This script demonstrates comprehensive exploratory data analysis techniques including:
- Data overview and basic statistics
- Missing value analysis
- Distribution analysis
- Correlation analysis
- Outlier detection
- Feature relationships
- Data visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_dataset():
    """Create a comprehensive sample dataset for EDA"""
    n_samples = 1000
    
    # Generate correlated features
    np.random.seed(42)
    age = np.random.normal(35, 10, n_samples)
    income = 50000 + age * 1000 + np.random.normal(0, 5000, n_samples)
    education_years = np.random.choice([12, 16, 18, 22], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    data = {
        'age': age,
        'income': income,
        'education_years': education_years,
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'satisfaction_score': np.random.randint(1, 11, n_samples),
        'purchase_amount': np.random.normal(100, 30, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'years_employed': np.random.exponential(5, n_samples),
        'has_children': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 30), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'credit_score'] = np.nan
    
    # Add some outliers
    df.loc[0, 'income'] = 500000  # Outlier
    df.loc[1, 'age'] = 150  # Outlier
    
    return df

def data_overview(df):
    """Provide a comprehensive overview of the dataset"""
    print("=== DATASET OVERVIEW ===\n")
    
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n=== COLUMN INFORMATION ===")
    print(df.info())
    
    print("\n=== BASIC STATISTICS ===")
    print(df.describe())
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())
    
    return df

def missing_value_analysis(df):
    """Analyze missing values in the dataset"""
    print("=== MISSING VALUE ANALYSIS ===\n")
    
    # Calculate missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percent
    }).sort_values('Missing_Count', ascending=False)
    
    print("Missing values summary:")
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    missing_df[missing_df['Missing_Count'] > 0]['Missing_Count'].plot(kind='bar')
    plt.title('Missing Values Count')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    missing_df[missing_df['Missing_Count'] > 0]['Missing_Percentage'].plot(kind='bar')
    plt.title('Missing Values Percentage')
    plt.xticks(rotation=45)
    plt.ylabel('Percentage (%)')
    
    plt.tight_layout()
    plt.show()
    
    return missing_df

def distribution_analysis(df):
    """Analyze distributions of numerical variables"""
    print("=== DISTRIBUTION ANALYSIS ===\n")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Numerical columns: {numerical_cols}")
    
    # Create distribution plots
    n_cols = len(numerical_cols)
    fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        # Histogram
        axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        axes[i].legend()
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Skewness and kurtosis analysis
    print("\nSkewness and Kurtosis Analysis:")
    skewness = df[numerical_cols].skew()
    kurtosis = df[numerical_cols].kurtosis()
    
    dist_stats = pd.DataFrame({
        'Skewness': skewness,
        'Kurtosis': kurtosis
    })
    print(dist_stats)

def correlation_analysis(df):
    """Analyze correlations between variables"""
    print("=== CORRELATION ANALYSIS ===\n")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated features
    high_corr = np.where(np.abs(correlation_matrix) > 0.7)
    high_corr = [(correlation_matrix.index[x], correlation_matrix.columns[y], correlation_matrix.iloc[x, y])
                 for x, y in zip(*high_corr) if x != y and x < y]
    
    if high_corr:
        print("\nHighly correlated features (|correlation| > 0.7):")
        for feat1, feat2, corr in high_corr:
            print(f"{feat1} - {feat2}: {corr:.3f}")
    else:
        print("\nNo highly correlated features found.")
    
    return correlation_matrix

def outlier_analysis(df):
    """Detect and analyze outliers"""
    print("=== OUTLIER ANALYSIS ===\n")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # IQR method for outlier detection
    outlier_info = {}
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outlier_info[col] = {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    # Create outlier summary
    outlier_df = pd.DataFrame(outlier_info).T
    print("Outlier Summary (IQR method):")
    print(outlier_df.round(3))
    
    # Visualize outliers with box plots
    fig, axes = plt.subplots(2, (len(numerical_cols) + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numerical_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'Box Plot - {col}')
        axes[i].set_ylabel(col)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return outlier_df

def categorical_analysis(df):
    """Analyze categorical variables"""
    print("=== CATEGORICAL VARIABLE ANALYSIS ===\n")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        print("No categorical variables found.")
        return
    
    print(f"Categorical columns: {categorical_cols}")
    
    # Analyze each categorical variable
    for col in categorical_cols:
        print(f"\n--- {col} ---")
        value_counts = df[col].value_counts()
        print("Value counts:")
        print(value_counts)
        print(f"\nPercentage distribution:")
        print((value_counts / len(df) * 100).round(2))
        
        # Visualize
        plt.figure(figsize=(10, 6))
        value_counts.plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def feature_relationships(df):
    """Analyze relationships between features"""
    print("=== FEATURE RELATIONSHIPS ===\n")
    
    # Scatter plots for numerical variables
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) >= 2:
        # Create scatter plot matrix
        fig, axes = plt.subplots(len(numerical_cols), len(numerical_cols), figsize=(15, 15))
        
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols):
                if i == j:
                    axes[i, j].hist(df[col1].dropna(), bins=20)
                    axes[i, j].set_title(col1)
                else:
                    axes[i, j].scatter(df[col1], df[col2], alpha=0.5)
                    axes[i, j].set_xlabel(col1)
                    axes[i, j].set_ylabel(col2)
        
        plt.tight_layout()
        plt.show()
    
    # Categorical vs Numerical analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols and numerical_cols:
        for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            for num_col in numerical_cols[:3]:  # Limit to first 3 numerical columns
                plt.figure(figsize=(10, 6))
                df.boxplot(column=num_col, by=cat_col)
                plt.title(f'{num_col} by {cat_col}')
                plt.suptitle('')  # Remove default title
                plt.tight_layout()
                plt.show()

def summary_statistics(df):
    """Generate comprehensive summary statistics"""
    print("=== SUMMARY STATISTICS ===\n")
    
    # Basic statistics
    print("Basic Statistics:")
    print(df.describe())
    
    # Memory usage
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Unique values per column
    print("\nUnique values per column:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

def main():
    """Main function to run the complete EDA pipeline"""
    print("=== EXPLORATORY DATA ANALYSIS PIPELINE ===\n")
    
    # 1. Create sample dataset
    print("1. Creating sample dataset...")
    df = create_sample_dataset()
    print("Dataset created successfully!")
    print("\n" + "="*60 + "\n")
    
    # 2. Data overview
    print("2. Data overview...")
    data_overview(df)
    print("\n" + "="*60 + "\n")
    
    # 3. Missing value analysis
    print("3. Missing value analysis...")
    missing_value_analysis(df)
    print("\n" + "="*60 + "\n")
    
    # 4. Distribution analysis
    print("4. Distribution analysis...")
    distribution_analysis(df)
    print("\n" + "="*60 + "\n")
    
    # 5. Correlation analysis
    print("5. Correlation analysis...")
    correlation_analysis(df)
    print("\n" + "="*60 + "\n")
    
    # 6. Outlier analysis
    print("6. Outlier analysis...")
    outlier_analysis(df)
    print("\n" + "="*60 + "\n")
    
    # 7. Categorical analysis
    print("7. Categorical analysis...")
    categorical_analysis(df)
    print("\n" + "="*60 + "\n")
    
    # 8. Feature relationships
    print("8. Feature relationships...")
    feature_relationships(df)
    print("\n" + "="*60 + "\n")
    
    # 9. Summary statistics
    print("9. Summary statistics...")
    summary_statistics(df)
    
    print("\n=== EDA COMPLETE! ===")
    print("The exploratory data analysis has been completed successfully!")

if __name__ == "__main__":
    main() 