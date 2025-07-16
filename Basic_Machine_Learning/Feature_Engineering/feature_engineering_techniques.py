"""
Feature Engineering Techniques
=============================

This script demonstrates various feature engineering techniques including:
- Feature creation and transformation
- Polynomial features
- Interaction features
- Time-based features
- Domain-specific features
- Feature selection methods
- Dimensionality reduction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_sample_dataset():
    """Create a comprehensive sample dataset for feature engineering"""
    n_samples = 1000
    
    # Generate base features
    np.random.seed(42)
    age = np.random.normal(35, 10, n_samples)
    income = 50000 + age * 1000 + np.random.normal(0, 5000, n_samples)
    education_years = np.random.choice([12, 16, 18, 22], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
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
        'has_children': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'registration_date': dates,
        'last_purchase_date': [d + timedelta(days=np.random.randint(0, 30)) for d in dates],
        'num_purchases': np.random.poisson(5, n_samples),
        'avg_purchase_value': np.random.normal(80, 20, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 30), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 20), 'credit_score'] = np.nan
    
    return df

def basic_feature_creation(df):
    """Create basic derived features"""
    print("=== BASIC FEATURE CREATION ===\n")
    
    # Mathematical transformations
    df['age_squared'] = df['age'] ** 2
    df['income_per_age'] = df['income'] / df['age']
    df['education_income_ratio'] = df['education_years'] / df['income'] * 10000
    
    # Binning features
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                            labels=['Young', 'Adult', 'Middle', 'Senior'])
    df['income_level'] = pd.cut(df['income'], bins=[0, 50000, 100000, 200000, float('inf')], 
                               labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Boolean features
    df['high_income'] = (df['income'] > df['income'].quantile(0.75)).astype(int)
    df['high_education'] = (df['education_years'] >= 16).astype(int)
    df['senior_citizen'] = (df['age'] >= 65).astype(int)
    
    # Ratio features
    df['purchase_frequency'] = df['num_purchases'] / (df['years_employed'] + 1)
    df['avg_purchase_to_income'] = df['avg_purchase_value'] / df['income'] * 100
    
    print("Basic features created:")
    new_features = ['age_squared', 'income_per_age', 'education_income_ratio', 
                   'age_group', 'income_level', 'high_income', 'high_education', 
                   'senior_citizen', 'purchase_frequency', 'avg_purchase_to_income']
    
    for feature in new_features:
        print(f"- {feature}")
    
    return df, new_features

def polynomial_features_creation(df, numerical_features):
    """Create polynomial features"""
    print("\n=== POLYNOMIAL FEATURES ===\n")
    
    # Select numerical features for polynomial transformation
    X_poly = df[numerical_features].fillna(df[numerical_features].mean())
    
    # Create polynomial features (degree=2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X_poly)
    
    # Create feature names
    feature_names = poly.get_feature_names_out(numerical_features)
    
    # Create DataFrame with polynomial features
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Add polynomial features to original dataframe
    for col in poly_df.columns:
        if col not in df.columns:  # Only add new features
            df[col] = poly_df[col]
    
    print(f"Polynomial features created (degree=2): {len(poly_df.columns)} features")
    print("Sample polynomial features:")
    for i, feature in enumerate(poly_df.columns[:10]):  # Show first 10
        print(f"- {feature}")
    
    return df, poly_df.columns.tolist()

def interaction_features(df, feature_pairs):
    """Create interaction features between specified pairs"""
    print("\n=== INTERACTION FEATURES ===\n")
    
    interaction_features = []
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            interaction_features.append(interaction_name)
            print(f"Created: {interaction_name}")
    
    return df, interaction_features

def time_based_features(df):
    """Create time-based features"""
    print("\n=== TIME-BASED FEATURES ===\n")
    
    # Convert date columns to datetime if they're not already
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
    
    # Extract time components
    df['registration_year'] = df['registration_date'].dt.year
    df['registration_month'] = df['registration_date'].dt.month
    df['registration_day'] = df['registration_date'].dt.day
    df['registration_dayofweek'] = df['registration_date'].dt.dayofweek
    
    # Time differences
    df['days_since_registration'] = (datetime.now() - df['registration_date']).dt.days
    df['days_since_last_purchase'] = (datetime.now() - df['last_purchase_date']).dt.days
    
    # Seasonal features
    df['is_weekend_registration'] = df['registration_dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['registration_day'].isin([1, 2, 3]).astype(int)
    df['is_month_end'] = df['registration_day'].isin([28, 29, 30, 31]).astype(int)
    
    # Customer lifetime features
    df['customer_lifetime_days'] = (df['last_purchase_date'] - df['registration_date']).dt.days
    df['purchase_rate'] = df['num_purchases'] / (df['customer_lifetime_days'] + 1)
    
    time_features = ['registration_year', 'registration_month', 'registration_day', 
                    'registration_dayofweek', 'days_since_registration', 
                    'days_since_last_purchase', 'is_weekend_registration', 
                    'is_month_start', 'is_month_end', 'customer_lifetime_days', 
                    'purchase_rate']
    
    print("Time-based features created:")
    for feature in time_features:
        print(f"- {feature}")
    
    return df, time_features

def domain_specific_features(df):
    """Create domain-specific features for customer analysis"""
    print("\n=== DOMAIN-SPECIFIC FEATURES ===\n")
    
    # Customer segmentation features
    df['total_spent'] = df['num_purchases'] * df['avg_purchase_value']
    df['customer_value'] = df['total_spent'] * df['satisfaction_score'] / 10
    
    # Risk features
    df['credit_risk'] = (df['credit_score'] < 600).astype(int)
    df['income_risk'] = (df['income'] < df['income'].quantile(0.25)).astype(int)
    
    # Engagement features
    df['engagement_score'] = (df['satisfaction_score'] + df['num_purchases'] * 2) / 3
    df['loyalty_score'] = df['customer_lifetime_days'] * df['satisfaction_score'] / 1000
    
    # Behavioral features
    df['high_value_customer'] = (df['customer_value'] > df['customer_value'].quantile(0.8)).astype(int)
    df['frequent_buyer'] = (df['num_purchases'] > df['num_purchases'].quantile(0.75)).astype(int)
    
    # Composite features
    df['customer_segment'] = pd.cut(df['customer_value'], 
                                   bins=[0, df['customer_value'].quantile(0.33), 
                                        df['customer_value'].quantile(0.67), float('inf')],
                                   labels=['Bronze', 'Silver', 'Gold'])
    
    domain_features = ['total_spent', 'customer_value', 'credit_risk', 'income_risk',
                      'engagement_score', 'loyalty_score', 'high_value_customer', 
                      'frequent_buyer', 'customer_segment']
    
    print("Domain-specific features created:")
    for feature in domain_features:
        print(f"- {feature}")
    
    return df, domain_features

def feature_selection_methods(df, target_column):
    """Demonstrate various feature selection methods"""
    print("\n=== FEATURE SELECTION METHODS ===\n")
    
    # Prepare data for feature selection
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    
    X = df[numerical_features].fillna(df[numerical_features].mean())
    y = df[target_column]
    
    # 1. Correlation-based selection
    print("1. Correlation-based selection:")
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    high_corr_features = correlations[correlations > 0.3].index.tolist()
    print(f"Features with correlation > 0.3: {len(high_corr_features)}")
    print(high_corr_features[:5])  # Show top 5
    
    # 2. Statistical tests (F-test)
    print("\n2. Statistical tests (F-test):")
    selector = SelectKBest(score_func=f_regression, k=10)
    selector.fit(X, y)
    f_test_features = X.columns[selector.get_support()].tolist()
    print(f"Top 10 features by F-test: {f_test_features}")
    
    # 3. Recursive Feature Elimination (RFE)
    print("\n3. Recursive Feature Elimination (RFE):")
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=10)
    rfe.fit(X, y)
    rfe_features = X.columns[rfe.support_].tolist()
    print(f"Top 10 features by RFE: {rfe_features}")
    
    # 4. Feature importance from Random Forest
    print("\n4. Random Forest feature importance:")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 features by Random Forest importance:")
    print(feature_importance.head(10))
    
    return {
        'correlation': high_corr_features,
        'f_test': f_test_features,
        'rfe': rfe_features,
        'random_forest': feature_importance.head(10)['feature'].tolist()
    }

def dimensionality_reduction(df, numerical_features, n_components=5):
    """Demonstrate dimensionality reduction techniques"""
    print("\n=== DIMENSIONALITY REDUCTION ===\n")
    
    # Prepare data
    X = df[numerical_features].fillna(df[numerical_features].mean())
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    print("1. Principal Component Analysis (PCA):")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Create DataFrame with PCA components
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
    
    # Add PCA components to original dataframe
    for col in pca_columns:
        df[col] = df_pca[col]
    
    print(f"PCA components added: {pca_columns}")
    
    return df, pca_columns

def feature_engineering_pipeline(df):
    """Complete feature engineering pipeline"""
    print("=== FEATURE ENGINEERING PIPELINE ===\n")
    
    original_features = df.columns.tolist()
    print(f"Original features: {len(original_features)}")
    
    # 1. Basic feature creation
    df, basic_features = basic_feature_creation(df)
    
    # 2. Polynomial features
    numerical_features = ['age', 'income', 'education_years', 'satisfaction_score', 'credit_score']
    df, poly_features = polynomial_features_creation(df, numerical_features)
    
    # 3. Interaction features
    interaction_pairs = [('age', 'income'), ('education_years', 'income'), 
                        ('satisfaction_score', 'credit_score')]
    df, interaction_features = interaction_features(df, interaction_pairs)
    
    # 4. Time-based features
    df, time_features = time_based_features(df)
    
    # 5. Domain-specific features
    df, domain_features = domain_specific_features(df)
    
    # 6. Feature selection
    selection_results = feature_selection_methods(df, 'purchase_amount')
    
    # 7. Dimensionality reduction
    all_numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    df, pca_features = dimensionality_reduction(df, all_numerical)
    
    # Summary
    final_features = df.columns.tolist()
    new_features = [f for f in final_features if f not in original_features]
    
    print(f"\n=== FEATURE ENGINEERING SUMMARY ===")
    print(f"Original features: {len(original_features)}")
    print(f"New features created: {len(new_features)}")
    print(f"Total features: {len(final_features)}")
    
    print(f"\nFeature categories:")
    print(f"- Basic features: {len(basic_features)}")
    print(f"- Polynomial features: {len(poly_features)}")
    print(f"- Interaction features: {len(interaction_features)}")
    print(f"- Time-based features: {len(time_features)}")
    print(f"- Domain-specific features: {len(domain_features)}")
    print(f"- PCA components: {len(pca_features)}")
    
    return df, {
        'original_features': original_features,
        'new_features': new_features,
        'basic_features': basic_features,
        'poly_features': poly_features,
        'interaction_features': interaction_features,
        'time_features': time_features,
        'domain_features': domain_features,
        'pca_features': pca_features,
        'selection_results': selection_results
    }

def main():
    """Main function to run the complete feature engineering pipeline"""
    print("=== FEATURE ENGINEERING TECHNIQUES DEMONSTRATION ===\n")
    
    # 1. Create sample dataset
    print("1. Creating sample dataset...")
    df = create_sample_dataset()
    print(f"Dataset created with shape: {df.shape}")
    print("\n" + "="*60 + "\n")
    
    # 2. Run complete feature engineering pipeline
    print("2. Running feature engineering pipeline...")
    df_engineered, feature_summary = feature_engineering_pipeline(df)
    
    print("\n" + "="*60 + "\n")
    print("=== FEATURE ENGINEERING COMPLETE! ===")
    print("The dataset has been successfully enhanced with engineered features!")
    print(f"Final dataset shape: {df_engineered.shape}")

if __name__ == "__main__":
    main() 