"""
Classification Problems Examples
================================

- Customer Churn Prediction
- Fraud Detection
- Disease Diagnosis
- Spam Detection
- Model comparison and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Customer Churn Prediction
print("=== Customer Churn Prediction ===")

def generate_churn_data(n_customers=1000):
    """Generate synthetic customer churn data"""
    np.random.seed(42)
    
    # Generate customer features
    tenure = np.random.exponential(scale=20, size=n_customers).astype(int)
    monthly_charges = np.random.normal(65, 20, n_customers)
    total_charges = monthly_charges * tenure + np.random.normal(0, 100, n_customers)
    
    # Generate categorical features
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.3, 0.4, 0.3])
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.5, 0.3, 0.2])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                    n_customers, p=[0.3, 0.2, 0.25, 0.25])
    
    # Generate churn based on features (higher churn for certain conditions)
    churn_prob = 0.1  # Base churn rate
    
    # Higher churn for month-to-month contracts
    churn_prob += np.where(contract == 'Month-to-month', 0.2, 0)
    
    # Higher churn for electronic check payment
    churn_prob += np.where(payment_method == 'Electronic check', 0.1, 0)
    
    # Higher churn for high monthly charges
    churn_prob += np.where(monthly_charges > 80, 0.15, 0)
    
    # Lower churn for longer tenure
    churn_prob -= np.where(tenure > 30, 0.1, 0)
    
    # Add some randomness
    churn_prob += np.random.normal(0, 0.05, n_customers)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    churn = np.random.binomial(1, churn_prob)
    
    return pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'internet_service': internet_service,
        'contract': contract,
        'payment_method': payment_method,
        'churn': churn
    })

# Generate churn data
churn_data = generate_churn_data()
print(f"Churn data shape: {churn_data.shape}")
print(f"Churn rate: {churn_data['churn'].mean():.3f}")

# Prepare features
churn_features = ['tenure', 'monthly_charges', 'total_charges', 'internet_service', 'contract', 'payment_method']
X_churn = churn_data[churn_features].copy()
y_churn = churn_data['churn']

# Encode categorical variables
le_internet = LabelEncoder()
le_contract = LabelEncoder()
le_payment = LabelEncoder()

X_churn['internet_service'] = le_internet.fit_transform(X_churn['internet_service'])
X_churn['contract'] = le_contract.fit_transform(X_churn['contract'])
X_churn['payment_method'] = le_payment.fit_transform(X_churn['payment_method'])

# Split data
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

# Scale features
scaler_churn = StandardScaler()
X_train_scaled_churn = scaler_churn.fit_transform(X_train_churn)
X_test_scaled_churn = scaler_churn.transform(X_test_churn)

# Train multiple models
models_churn = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
}

churn_results = {}
for name, model in models_churn.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled_churn, y_train_churn)
    y_pred = model.predict(X_test_scaled_churn)
    y_pred_proba = model.predict_proba(X_test_scaled_churn)[:, 1]
    
    churn_results[name] = {
        'accuracy': accuracy_score(y_test_churn, y_pred),
        'precision': precision_score(y_test_churn, y_pred),
        'recall': recall_score(y_test_churn, y_pred),
        'f1': f1_score(y_test_churn, y_pred),
        'auc': roc_auc_score(y_test_churn, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# 2. Fraud Detection
print("\n=== Fraud Detection ===")

def generate_fraud_data(n_transactions=5000):
    """Generate synthetic credit card fraud data"""
    np.random.seed(42)
    
    # Generate normal transactions
    n_normal = int(n_transactions * 0.95)
    n_fraud = n_transactions - n_normal
    
    # Normal transaction features
    normal_amount = np.random.exponential(scale=50, size=n_normal)
    normal_hour = np.random.randint(0, 24, n_normal)
    normal_distance = np.random.exponential(scale=10, size=n_normal)
    
    # Fraud transaction features (different patterns)
    fraud_amount = np.random.exponential(scale=200, size=n_fraud)  # Higher amounts
    fraud_hour = np.random.choice([0, 1, 2, 3, 22, 23], size=n_fraud)  # Late night
    fraud_distance = np.random.exponential(scale=50, size=n_fraud)  # Higher distance
    
    # Combine data
    amounts = np.concatenate([normal_amount, fraud_amount])
    hours = np.concatenate([normal_hour, fraud_hour])
    distances = np.concatenate([normal_distance, fraud_distance])
    
    # Generate additional features
    merchant_category = np.random.randint(1, 21, n_transactions)
    card_type = np.random.choice(['Visa', 'Mastercard', 'Amex'], n_transactions)
    
    # Create fraud labels
    fraud = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle data
    indices = np.random.permutation(n_transactions)
    
    return pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'amount': amounts[indices],
        'hour': hours[indices],
        'distance': distances[indices],
        'merchant_category': merchant_category[indices],
        'card_type': card_type[indices],
        'fraud': fraud[indices]
    })

# Generate fraud data
fraud_data = generate_fraud_data()
print(f"Fraud data shape: {fraud_data.shape}")
print(f"Fraud rate: {fraud_data['fraud'].mean():.3f}")

# Prepare features
fraud_features = ['amount', 'hour', 'distance', 'merchant_category', 'card_type']
X_fraud = fraud_data[fraud_features].copy()
y_fraud = fraud_data['fraud']

# Encode categorical variables
le_card = LabelEncoder()
X_fraud['card_type'] = le_card.fit_transform(X_fraud['card_type'])

# Split data
X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = train_test_split(
    X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
)

# Scale features
scaler_fraud = StandardScaler()
X_train_scaled_fraud = scaler_fraud.fit_transform(X_train_fraud)
X_test_scaled_fraud = scaler_fraud.transform(X_test_fraud)

# Train models for fraud detection
fraud_results = {}
for name, model in models_churn.items():
    print(f"\nTraining {name} for fraud detection...")
    model.fit(X_train_scaled_fraud, y_train_fraud)
    y_pred = model.predict(X_test_scaled_fraud)
    y_pred_proba = model.predict_proba(X_test_scaled_fraud)[:, 1]
    
    fraud_results[name] = {
        'accuracy': accuracy_score(y_test_fraud, y_pred),
        'precision': precision_score(y_test_fraud, y_pred),
        'recall': recall_score(y_test_fraud, y_pred),
        'f1': f1_score(y_test_fraud, y_pred),
        'auc': roc_auc_score(y_test_fraud, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# 3. Disease Diagnosis
print("\n=== Disease Diagnosis ===")

def generate_diagnosis_data(n_patients=2000):
    """Generate synthetic medical diagnosis data"""
    np.random.seed(42)
    
    # Generate patient features
    age = np.random.normal(50, 15, n_patients)
    age = np.clip(age, 18, 90)
    
    bmi = np.random.normal(25, 5, n_patients)
    bmi = np.clip(bmi, 15, 45)
    
    blood_pressure = np.random.normal(120, 20, n_patients)
    blood_pressure = np.clip(blood_pressure, 80, 200)
    
    cholesterol = np.random.normal(200, 40, n_patients)
    cholesterol = np.clip(cholesterol, 100, 400)
    
    glucose = np.random.normal(100, 20, n_patients)
    glucose = np.clip(glucose, 60, 200)
    
    # Generate categorical features
    gender = np.random.choice(['Male', 'Female'], n_patients)
    smoking = np.random.choice(['Yes', 'No'], n_patients, p=[0.3, 0.7])
    diabetes = np.random.choice(['Yes', 'No'], n_patients, p=[0.2, 0.8])
    
    # Generate disease probability based on features
    disease_prob = 0.05  # Base disease rate
    
    # Age effect
    disease_prob += np.where(age > 60, 0.1, 0)
    
    # BMI effect
    disease_prob += np.where(bmi > 30, 0.15, 0)
    
    # Blood pressure effect
    disease_prob += np.where(blood_pressure > 140, 0.1, 0)
    
    # Cholesterol effect
    disease_prob += np.where(cholesterol > 240, 0.1, 0)
    
    # Glucose effect
    disease_prob += np.where(glucose > 126, 0.2, 0)
    
    # Smoking effect
    disease_prob += np.where(smoking == 'Yes', 0.1, 0)
    
    # Diabetes effect
    disease_prob += np.where(diabetes == 'Yes', 0.15, 0)
    
    # Add randomness
    disease_prob += np.random.normal(0, 0.05, n_patients)
    disease_prob = np.clip(disease_prob, 0, 1)
    
    disease = np.random.binomial(1, disease_prob)
    
    return pd.DataFrame({
        'patient_id': range(1, n_patients + 1),
        'age': age,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'gender': gender,
        'smoking': smoking,
        'diabetes': diabetes,
        'disease': disease
    })

# Generate diagnosis data
diagnosis_data = generate_diagnosis_data()
print(f"Diagnosis data shape: {diagnosis_data.shape}")
print(f"Disease rate: {diagnosis_data['disease'].mean():.3f}")

# Prepare features
diagnosis_features = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose', 'gender', 'smoking', 'diabetes']
X_diagnosis = diagnosis_data[diagnosis_features].copy()
y_diagnosis = diagnosis_data['disease']

# Encode categorical variables
le_gender = LabelEncoder()
le_smoking = LabelEncoder()
le_diabetes = LabelEncoder()

X_diagnosis['gender'] = le_gender.fit_transform(X_diagnosis['gender'])
X_diagnosis['smoking'] = le_smoking.fit_transform(X_diagnosis['smoking'])
X_diagnosis['diabetes'] = le_diabetes.fit_transform(X_diagnosis['diabetes'])

# Split data
X_train_diagnosis, X_test_diagnosis, y_train_diagnosis, y_test_diagnosis = train_test_split(
    X_diagnosis, y_diagnosis, test_size=0.2, random_state=42, stratify=y_diagnosis
)

# Scale features
scaler_diagnosis = StandardScaler()
X_train_scaled_diagnosis = scaler_diagnosis.fit_transform(X_train_diagnosis)
X_test_scaled_diagnosis = scaler_diagnosis.transform(X_test_diagnosis)

# Train models for diagnosis
diagnosis_results = {}
for name, model in models_churn.items():
    print(f"\nTraining {name} for disease diagnosis...")
    model.fit(X_train_scaled_diagnosis, y_train_diagnosis)
    y_pred = model.predict(X_test_scaled_diagnosis)
    y_pred_proba = model.predict_proba(X_test_scaled_diagnosis)[:, 1]
    
    diagnosis_results[name] = {
        'accuracy': accuracy_score(y_test_diagnosis, y_pred),
        'precision': precision_score(y_test_diagnosis, y_pred),
        'recall': recall_score(y_test_diagnosis, y_pred),
        'f1': f1_score(y_test_diagnosis, y_pred),
        'auc': roc_auc_score(y_test_diagnosis, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# 4. Spam Detection
print("\n=== Spam Detection ===")

def generate_spam_data(n_emails=3000):
    """Generate synthetic email spam data"""
    np.random.seed(42)
    
    # Generate email features
    word_count = np.random.exponential(scale=50, size=n_emails)
    word_count = np.clip(word_count, 10, 500)
    
    # Spam emails tend to be shorter
    spam_word_count = np.random.exponential(scale=30, size=n_emails)
    spam_word_count = np.clip(spam_word_count, 5, 200)
    
    # Combine and shuffle
    all_word_counts = np.concatenate([word_count, spam_word_count])
    np.random.shuffle(all_word_counts)
    
    # Generate other features
    capital_ratio = np.random.beta(2, 5, n_emails)  # Most emails have low capital ratio
    exclamation_count = np.random.poisson(1, n_emails)
    link_count = np.random.poisson(0.5, n_emails)
    
    # Spam emails have more capitals and exclamations
    spam_capital_ratio = np.random.beta(5, 2, n_emails)
    spam_exclamation_count = np.random.poisson(3, n_emails)
    spam_link_count = np.random.poisson(2, n_emails)
    
    # Combine features
    all_capital_ratios = np.concatenate([capital_ratio, spam_capital_ratio])
    all_exclamation_counts = np.concatenate([exclamation_count, spam_exclamation_count])
    all_link_counts = np.concatenate([link_count, spam_link_count])
    
    # Shuffle
    np.random.shuffle(all_capital_ratios)
    np.random.shuffle(all_exclamation_counts)
    np.random.shuffle(all_link_counts)
    
    # Generate categorical features
    sender_type = np.random.choice(['Known', 'Unknown'], n_emails, p=[0.7, 0.3])
    subject_type = np.random.choice(['Normal', 'Suspicious'], n_emails, p=[0.8, 0.2])
    
    # Generate spam labels
    spam = np.concatenate([np.zeros(n_emails), np.ones(n_emails)])
    np.random.shuffle(spam)
    
    return pd.DataFrame({
        'email_id': range(1, n_emails + 1),
        'word_count': all_word_counts,
        'capital_ratio': all_capital_ratios,
        'exclamation_count': all_exclamation_counts,
        'link_count': all_link_counts,
        'sender_type': sender_type,
        'subject_type': subject_type,
        'spam': spam
    })

# Generate spam data
spam_data = generate_spam_data()
print(f"Spam data shape: {spam_data.shape}")
print(f"Spam rate: {spam_data['spam'].mean():.3f}")

# Prepare features
spam_features = ['word_count', 'capital_ratio', 'exclamation_count', 'link_count', 'sender_type', 'subject_type']
X_spam = spam_data[spam_features].copy()
y_spam = spam_data['spam']

# Encode categorical variables
le_sender = LabelEncoder()
le_subject = LabelEncoder()

X_spam['sender_type'] = le_sender.fit_transform(X_spam['sender_type'])
X_spam['subject_type'] = le_subject.fit_transform(X_spam['subject_type'])

# Split data
X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
    X_spam, y_spam, test_size=0.2, random_state=42, stratify=y_spam
)

# Scale features
scaler_spam = StandardScaler()
X_train_scaled_spam = scaler_spam.fit_transform(X_train_spam)
X_test_scaled_spam = scaler_spam.transform(X_test_spam)

# Train models for spam detection
spam_results = {}
for name, model in models_churn.items():
    print(f"\nTraining {name} for spam detection...")
    model.fit(X_train_scaled_spam, y_train_spam)
    y_pred = model.predict(X_test_scaled_spam)
    y_pred_proba = model.predict_proba(X_test_scaled_spam)[:, 1]
    
    spam_results[name] = {
        'accuracy': accuracy_score(y_test_spam, y_pred),
        'precision': precision_score(y_test_spam, y_pred),
        'recall': recall_score(y_test_spam, y_pred),
        'f1': f1_score(y_test_spam, y_pred),
        'auc': roc_auc_score(y_test_spam, y_pred_proba),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# 5. Model Comparison and Visualization
print("\n=== Model Comparison and Visualization ===")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Function to plot results
def plot_results(results, title, ax):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    # Create data for plotting
    data = []
    for metric in metrics:
        values = [results[model][metric] for model in models]
        data.append(values)
    
    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.15
    
    for i, (metric, values) in enumerate(zip(metrics, data)):
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot results for each problem
plot_results(churn_results, 'Customer Churn Prediction', axes[0, 0])
plot_results(fraud_results, 'Fraud Detection', axes[0, 1])
plot_results(diagnosis_results, 'Disease Diagnosis', axes[0, 2])

# Plot ROC curves for best models
def plot_roc_curves(results, title, ax):
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test_churn, result['probabilities'])
        auc = result['auc']
        ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_roc_curves(churn_results, 'Churn Prediction', axes[1, 0])
plot_roc_curves(fraud_results, 'Fraud Detection', axes[1, 1])
plot_roc_curves(diagnosis_results, 'Disease Diagnosis', axes[1, 2])

plt.tight_layout()
plt.show()

# 6. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# Get feature importance from Random Forest models
rf_churn = models_churn['Random Forest']
rf_fraud = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled_fraud, y_train_fraud)
rf_diagnosis = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled_diagnosis, y_train_diagnosis)
rf_spam = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled_spam, y_train_spam)

# Plot feature importance
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Churn feature importance
churn_importance = pd.DataFrame({
    'feature': churn_features,
    'importance': rf_churn.feature_importances_
}).sort_values('importance', ascending=True)

axes[0, 0].barh(churn_importance['feature'], churn_importance['importance'])
axes[0, 0].set_title('Feature Importance - Churn Prediction')
axes[0, 0].set_xlabel('Importance')

# Fraud feature importance
fraud_importance = pd.DataFrame({
    'feature': fraud_features,
    'importance': rf_fraud.feature_importances_
}).sort_values('importance', ascending=True)

axes[0, 1].barh(fraud_importance['feature'], fraud_importance['importance'])
axes[0, 1].set_title('Feature Importance - Fraud Detection')
axes[0, 1].set_xlabel('Importance')

# Diagnosis feature importance
diagnosis_importance = pd.DataFrame({
    'feature': diagnosis_features,
    'importance': rf_diagnosis.feature_importances_
}).sort_values('importance', ascending=True)

axes[1, 0].barh(diagnosis_importance['feature'], diagnosis_importance['importance'])
axes[1, 0].set_title('Feature Importance - Disease Diagnosis')
axes[1, 0].set_xlabel('Importance')

# Spam feature importance
spam_importance = pd.DataFrame({
    'feature': spam_features,
    'importance': rf_spam.feature_importances_
}).sort_values('importance', ascending=True)

axes[1, 1].barh(spam_importance['feature'], spam_importance['importance'])
axes[1, 1].set_title('Feature Importance - Spam Detection')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# 7. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

# Find best model for each problem
def find_best_model(results, metric='auc'):
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1][metric]

print("Best Models by AUC Score:")
print(f"1. Customer Churn Prediction:")
best_churn, churn_auc = find_best_model(churn_results)
print(f"   Best model: {best_churn} (AUC: {churn_auc:.3f})")

print(f"\n2. Fraud Detection:")
best_fraud, fraud_auc = find_best_model(fraud_results)
print(f"   Best model: {best_fraud} (AUC: {fraud_auc:.3f})")

print(f"\n3. Disease Diagnosis:")
best_diagnosis, diagnosis_auc = find_best_model(diagnosis_results)
print(f"   Best model: {best_diagnosis} (AUC: {diagnosis_auc:.3f})")

print(f"\n4. Spam Detection:")
best_spam, spam_auc = find_best_model(spam_results)
print(f"   Best model: {best_spam} (AUC: {spam_auc:.3f})")

print(f"\nKey Insights:")
print(f"- Random Forest and Gradient Boosting perform well across all problems")
print(f"- Neural Networks show good performance for complex patterns")
print(f"- Logistic Regression is interpretable but may underperform on complex data")
print(f"- Model choice depends on data characteristics and business requirements")

print(f"\nRecommendations:")
print(f"- Use Random Forest for balanced performance and interpretability")
print(f"- Use Gradient Boosting for maximum predictive performance")
print(f"- Use Logistic Regression when interpretability is crucial")
print(f"- Always consider class imbalance in fraud and disease detection")
print(f"- Feature engineering is crucial for all classification problems") 