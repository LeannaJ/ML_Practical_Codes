"""
Basic Data Preprocessing
=======================

This script demonstrates fundamental data preprocessing techniques including:
- Handling missing values
- Data type conversion
- Scaling and normalization
- Encoding categorical variables
- Feature selection basics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_dataset():
    """Create sample dataset with various data types and missing values"""
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], n_samples),
        'satisfaction_score': np.random.randint(1, 11, n_samples),
        'purchase_amount': np.random.normal(100, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'education'] = np.nan
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("Before handling missing values:")
    print(df.isnull().sum())
    
    # For numerical columns, use mean imputation
    numerical_imputer = SimpleImputer(strategy='mean')
    df[['age', 'income']] = numerical_imputer.fit_transform(df[['age', 'income']])
    
    # For categorical columns, use mode imputation
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[['education']] = categorical_imputer.fit_transform(df[['education']])
    
    print("\nAfter handling missing values:")
    print(df.isnull().sum())
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables"""
    # Label encoding for ordinal categorical variables
    label_encoder = LabelEncoder()
    df['education_encoded'] = label_encoder.fit_transform(df['education'])
    
    print("Education encoding mapping:")
    for i, category in enumerate(label_encoder.classes_):
        print(f"{category}: {i}")
    
    # One-hot encoding for nominal categorical variables
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')
    city_encoded = onehot_encoder.fit_transform(df[['city']])
    city_columns = [f"city_{city}" for city in onehot_encoder.categories_[0][1:]]
    df_city_encoded = pd.DataFrame(city_encoded, columns=city_columns, index=df.index)
    
    df = pd.concat([df, df_city_encoded], axis=1)
    
    return df, city_columns

def scale_features(df, numerical_features):
    """Scale numerical features"""
    # Standard scaling (z-score normalization)
    standard_scaler = StandardScaler()
    df_scaled_standard = df.copy()
    df_scaled_standard[numerical_features] = standard_scaler.fit_transform(df[numerical_features])
    
    # Min-Max scaling (normalization to [0,1])
    minmax_scaler = MinMaxScaler()
    df_scaled_minmax = df.copy()
    df_scaled_minmax[numerical_features] = minmax_scaler.fit_transform(df[numerical_features])
    
    print("Original data statistics:")
    print(df[numerical_features].describe())
    
    print("\nStandard scaled data statistics:")
    print(df_scaled_standard[numerical_features].describe())
    
    print("\nMin-Max scaled data statistics:")
    print(df_scaled_minmax[numerical_features].describe())
    
    return df_scaled_standard, df_scaled_minmax

def convert_data_types(df):
    """Convert data types appropriately"""
    df['age'] = df['age'].astype(int)
    df['satisfaction_score'] = df['satisfaction_score'].astype(int)
    df['income'] = df['income'].astype(float)
    
    print("Data types after conversion:")
    print(df.dtypes)
    
    return df

def engineer_features(df):
    """Create new features"""
    df['income_per_age'] = df['income'] / df['age']
    df['high_income'] = (df['income'] > df['income'].quantile(0.75)).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior'])
    
    print("New features created:")
    print(df[['income_per_age', 'high_income', 'age_group']].head())
    
    return df

def prepare_final_dataset(df, city_columns):
    """Prepare final dataset for modeling"""
    # Select final features for modeling
    feature_columns = ['age', 'income', 'satisfaction_score', 'purchase_amount', 
                      'education_encoded', 'income_per_age', 'high_income'] + city_columns
    
    X = df[feature_columns]
    y = df['purchase_amount']  # Target variable
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Final dataset shape:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    print("\nFeature columns:")
    print(X.columns.tolist())
    
    return X_train, X_test, y_train, y_test

def main():
    """Main function to run the complete preprocessing pipeline"""
    print("=== Basic Data Preprocessing Pipeline ===\n")
    
    # 1. Create sample dataset
    print("1. Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset shape: {df.shape}")
    print("First few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")
    
    # 2. Handle missing values
    print("2. Handling missing values...")
    df = handle_missing_values(df)
    print("\n" + "="*50 + "\n")
    
    # 3. Encode categorical variables
    print("3. Encoding categorical variables...")
    df, city_columns = encode_categorical_variables(df)
    print("\n" + "="*50 + "\n")
    
    # 4. Scale features
    print("4. Scaling features...")
    numerical_features = ['age', 'income', 'satisfaction_score', 'purchase_amount']
    df_scaled_standard, df_scaled_minmax = scale_features(df, numerical_features)
    print("\n" + "="*50 + "\n")
    
    # 5. Convert data types
    print("5. Converting data types...")
    df = convert_data_types(df)
    print("\n" + "="*50 + "\n")
    
    # 6. Engineer features
    print("6. Engineering features...")
    df = engineer_features(df)
    print("\n" + "="*50 + "\n")
    
    # 7. Prepare final dataset
    print("7. Preparing final dataset...")
    X_train, X_test, y_train, y_test = prepare_final_dataset(df, city_columns)
    
    print("\n=== Preprocessing Complete! ===")
    print("The preprocessed data is now ready for machine learning model training!")

if __name__ == "__main__":
    main() 