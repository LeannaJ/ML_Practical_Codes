"""
Anomaly Detection Examples
==========================

- Network Intrusion Detection
- Credit Card Fraud Detection
- Manufacturing Quality Control
- Time Series Anomaly Detection
- Model comparison and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# For deep learning-based anomaly detection
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# 1. Network Intrusion Detection
print("=== Network Intrusion Detection ===")

def generate_network_data(n_connections=10000, anomaly_ratio=0.1):
    """Generate synthetic network connection data"""
    np.random.seed(42)
    
    n_anomalies = int(n_connections * anomaly_ratio)
    n_normal = n_connections - n_anomalies
    
    # Normal connection features
    normal_duration = np.random.exponential(scale=100, size=n_normal)
    normal_src_bytes = np.random.exponential(scale=1000, size=n_normal)
    normal_dst_bytes = np.random.exponential(scale=500, size=n_normal)
    normal_count = np.random.poisson(lam=5, size=n_normal)
    normal_srv_count = np.random.poisson(lam=3, size=n_normal)
    normal_serror_rate = np.random.beta(1, 10, size=n_normal)
    normal_srv_serror_rate = np.random.beta(1, 10, size=n_normal)
    normal_rerror_rate = np.random.beta(1, 10, size=n_normal)
    normal_srv_rerror_rate = np.random.beta(1, 10, size=n_normal)
    
    # Anomalous connection features (different patterns)
    anomaly_duration = np.random.exponential(scale=500, size=n_anomalies)  # Longer duration
    anomaly_src_bytes = np.random.exponential(scale=5000, size=n_anomalies)  # More bytes
    anomaly_dst_bytes = np.random.exponential(scale=100, size=n_anomalies)  # Fewer dst bytes
    anomaly_count = np.random.poisson(lam=20, size=n_anomalies)  # Higher count
    anomaly_srv_count = np.random.poisson(lam=1, size=n_anomalies)  # Lower srv count
    anomaly_serror_rate = np.random.beta(5, 1, size=n_anomalies)  # Higher error rates
    anomaly_srv_serror_rate = np.random.beta(5, 1, size=n_anomalies)
    anomaly_rerror_rate = np.random.beta(5, 1, size=n_anomalies)
    anomaly_srv_rerror_rate = np.random.beta(5, 1, size=n_anomalies)
    
    # Combine data
    duration = np.concatenate([normal_duration, anomaly_duration])
    src_bytes = np.concatenate([normal_src_bytes, anomaly_src_bytes])
    dst_bytes = np.concatenate([normal_dst_bytes, anomaly_dst_bytes])
    count = np.concatenate([normal_count, anomaly_count])
    srv_count = np.concatenate([normal_srv_count, anomaly_srv_count])
    serror_rate = np.concatenate([normal_serror_rate, anomaly_serror_rate])
    srv_serror_rate = np.concatenate([normal_srv_serror_rate, anomaly_srv_serror_rate])
    rerror_rate = np.concatenate([normal_rerror_rate, anomaly_rerror_rate])
    srv_rerror_rate = np.concatenate([normal_srv_rerror_rate, anomaly_srv_rerror_rate])
    
    # Generate categorical features
    protocol_type = np.random.choice(['tcp', 'udp', 'icmp'], n_connections, p=[0.7, 0.2, 0.1])
    service = np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns'], n_connections, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    flag = np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], n_connections, p=[0.6, 0.2, 0.1, 0.1])
    
    # Create labels
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle data
    indices = np.random.permutation(n_connections)
    
    return pd.DataFrame({
        'connection_id': range(1, n_connections + 1),
        'duration': duration[indices],
        'src_bytes': src_bytes[indices],
        'dst_bytes': dst_bytes[indices],
        'count': count[indices],
        'srv_count': srv_count[indices],
        'serror_rate': serror_rate[indices],
        'srv_serror_rate': srv_serror_rate[indices],
        'rerror_rate': rerror_rate[indices],
        'srv_rerror_rate': srv_rerror_rate[indices],
        'protocol_type': protocol_type[indices],
        'service': service[indices],
        'flag': flag[indices],
        'anomaly': labels[indices]
    })

# Generate network data
network_data = generate_network_data()
print(f"Network data shape: {network_data.shape}")
print(f"Anomaly rate: {network_data['anomaly'].mean():.3f}")

# Prepare features
network_features = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count', 
                   'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                   'protocol_type', 'service', 'flag']
X_network = network_data[network_features].copy()
y_network = network_data['anomaly']

# Encode categorical variables
le_protocol = LabelEncoder()
le_service = LabelEncoder()
le_flag = LabelEncoder()

X_network['protocol_type'] = le_protocol.fit_transform(X_network['protocol_type'])
X_network['service'] = le_service.fit_transform(X_network['service'])
X_network['flag'] = le_flag.fit_transform(X_network['flag'])

# Split data
X_train_network, X_test_network, y_train_network, y_test_network = train_test_split(
    X_network, y_network, test_size=0.2, random_state=42, stratify=y_network
)

# Scale features
scaler_network = StandardScaler()
X_train_scaled_network = scaler_network.fit_transform(X_train_network)
X_test_scaled_network = scaler_network.transform(X_test_network)

# 2. Credit Card Fraud Detection
print("\n=== Credit Card Fraud Detection ===")

def generate_credit_card_data(n_transactions=50000, fraud_ratio=0.01):
    """Generate synthetic credit card transaction data"""
    np.random.seed(42)
    
    n_fraud = int(n_transactions * fraud_ratio)
    n_normal = n_transactions - n_fraud
    
    # Normal transaction features
    normal_amount = np.random.exponential(scale=50, size=n_normal)
    normal_hour = np.random.randint(0, 24, size=n_normal)
    normal_distance = np.random.exponential(scale=10, size=n_normal)
    normal_merchant_category = np.random.randint(1, 21, size=n_normal)
    
    # Fraud transaction features (different patterns)
    fraud_amount = np.random.exponential(scale=200, size=n_fraud)  # Higher amounts
    fraud_hour = np.random.choice([0, 1, 2, 3, 22, 23], size=n_fraud)  # Late night
    fraud_distance = np.random.exponential(scale=50, size=n_fraud)  # Higher distance
    fraud_merchant_category = np.random.choice([1, 2, 3, 4, 5], size=n_fraud)  # Specific categories
    
    # Combine data
    amounts = np.concatenate([normal_amount, fraud_amount])
    hours = np.concatenate([normal_hour, fraud_hour])
    distances = np.concatenate([normal_distance, fraud_distance])
    merchant_categories = np.concatenate([normal_merchant_category, fraud_merchant_category])
    
    # Generate additional features
    card_type = np.random.choice(['Visa', 'Mastercard', 'Amex'], n_transactions)
    transaction_type = np.random.choice(['Online', 'In-store', 'ATM'], n_transactions, p=[0.6, 0.3, 0.1])
    
    # Create fraud labels
    fraud = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Shuffle data
    indices = np.random.permutation(n_transactions)
    
    return pd.DataFrame({
        'transaction_id': range(1, n_transactions + 1),
        'amount': amounts[indices],
        'hour': hours[indices],
        'distance': distances[indices],
        'merchant_category': merchant_categories[indices],
        'card_type': card_type[indices],
        'transaction_type': transaction_type[indices],
        'fraud': fraud[indices]
    })

# Generate credit card data
credit_data = generate_credit_card_data()
print(f"Credit card data shape: {credit_data.shape}")
print(f"Fraud rate: {credit_data['fraud'].mean():.3f}")

# Prepare features
credit_features = ['amount', 'hour', 'distance', 'merchant_category', 'card_type', 'transaction_type']
X_credit = credit_data[credit_features].copy()
y_credit = credit_data['fraud']

# Encode categorical variables
le_card_type = LabelEncoder()
le_transaction_type = LabelEncoder()

X_credit['card_type'] = le_card_type.fit_transform(X_credit['card_type'])
X_credit['transaction_type'] = le_transaction_type.fit_transform(X_credit['transaction_type'])

# Split data
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(
    X_credit, y_credit, test_size=0.2, random_state=42, stratify=y_credit
)

# Scale features
scaler_credit = StandardScaler()
X_train_scaled_credit = scaler_credit.fit_transform(X_train_credit)
X_test_scaled_credit = scaler_credit.transform(X_test_credit)

# 3. Manufacturing Quality Control
print("\n=== Manufacturing Quality Control ===")

def generate_manufacturing_data(n_samples=8000, defect_ratio=0.05):
    """Generate synthetic manufacturing quality control data"""
    np.random.seed(42)
    
    n_defects = int(n_samples * defect_ratio)
    n_normal = n_samples - n_defects
    
    # Normal product features
    normal_temperature = np.random.normal(25, 2, n_normal)
    normal_pressure = np.random.normal(100, 5, n_normal)
    normal_speed = np.random.normal(50, 3, n_normal)
    normal_vibration = np.random.normal(0.1, 0.02, n_normal)
    normal_humidity = np.random.normal(45, 5, n_normal)
    normal_ph = np.random.normal(7, 0.5, n_normal)
    
    # Defective product features (out of normal ranges)
    defect_temperature = np.random.normal(35, 3, n_defects)  # Higher temperature
    defect_pressure = np.random.normal(120, 8, n_defects)  # Higher pressure
    defect_speed = np.random.normal(30, 5, n_defects)  # Lower speed
    defect_vibration = np.random.normal(0.3, 0.05, n_defects)  # Higher vibration
    defect_humidity = np.random.normal(70, 8, n_defects)  # Higher humidity
    defect_ph = np.random.normal(4, 1, n_defects)  # Lower pH
    
    # Combine data
    temperature = np.concatenate([normal_temperature, defect_temperature])
    pressure = np.concatenate([normal_pressure, defect_pressure])
    speed = np.concatenate([normal_speed, defect_speed])
    vibration = np.concatenate([normal_vibration, defect_vibration])
    humidity = np.concatenate([normal_humidity, defect_humidity])
    ph = np.concatenate([normal_ph, defect_ph])
    
    # Generate categorical features
    machine_id = np.random.choice(['Machine_A', 'Machine_B', 'Machine_C'], n_samples, p=[0.4, 0.35, 0.25])
    shift = np.random.choice(['Morning', 'Afternoon', 'Night'], n_samples, p=[0.4, 0.4, 0.2])
    operator = np.random.choice(['Op_1', 'Op_2', 'Op_3', 'Op_4'], n_samples)
    
    # Create defect labels
    defect = np.concatenate([np.zeros(n_normal), np.ones(n_defects)])
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    
    return pd.DataFrame({
        'sample_id': range(1, n_samples + 1),
        'temperature': temperature[indices],
        'pressure': pressure[indices],
        'speed': speed[indices],
        'vibration': vibration[indices],
        'humidity': humidity[indices],
        'ph': ph[indices],
        'machine_id': machine_id[indices],
        'shift': shift[indices],
        'operator': operator[indices],
        'defect': defect[indices]
    })

# Generate manufacturing data
manufacturing_data = generate_manufacturing_data()
print(f"Manufacturing data shape: {manufacturing_data.shape}")
print(f"Defect rate: {manufacturing_data['defect'].mean():.3f}")

# Prepare features
manufacturing_features = ['temperature', 'pressure', 'speed', 'vibration', 'humidity', 'ph',
                         'machine_id', 'shift', 'operator']
X_manufacturing = manufacturing_data[manufacturing_features].copy()
y_manufacturing = manufacturing_data['defect']

# Encode categorical variables
le_machine = LabelEncoder()
le_shift = LabelEncoder()
le_operator = LabelEncoder()

X_manufacturing['machine_id'] = le_machine.fit_transform(X_manufacturing['machine_id'])
X_manufacturing['shift'] = le_shift.fit_transform(X_manufacturing['shift'])
X_manufacturing['operator'] = le_operator.fit_transform(X_manufacturing['operator'])

# Split data
X_train_manufacturing, X_test_manufacturing, y_train_manufacturing, y_test_manufacturing = train_test_split(
    X_manufacturing, y_manufacturing, test_size=0.2, random_state=42, stratify=y_manufacturing
)

# Scale features
scaler_manufacturing = StandardScaler()
X_train_scaled_manufacturing = scaler_manufacturing.fit_transform(X_train_manufacturing)
X_test_scaled_manufacturing = scaler_manufacturing.transform(X_test_manufacturing)

# 4. Anomaly Detection Models
print("\n=== Anomaly Detection Models ===")

# Unsupervised models (trained only on normal data)
def train_unsupervised_models(X_train_normal, X_test, y_test):
    """Train unsupervised anomaly detection models"""
    models = {
        'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
        'One-Class SVM': OneClassSVM(kernel='rbf', nu=0.1),
        'Local Outlier Factor': LocalOutlierFactor(contamination=0.1, novelty=True),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'DBSCAN':
            # DBSCAN doesn't have predict method, use fit_predict
            labels = model.fit_predict(X_train_normal)
            # Convert labels: -1 (anomaly) -> 1, others -> 0
            y_pred = (labels == -1).astype(int)
            # For evaluation, we need predictions for test set
            # This is a simplified approach - in practice, you'd need a different method
            y_pred_test = np.zeros(len(X_test))
        else:
            model.fit(X_train_normal)
            y_pred = model.predict(X_test)
            # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
            y_pred = (y_pred == -1).astype(int)
            y_pred_test = y_pred
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'predictions': y_pred_test
        }
    
    return results

# Supervised models (trained on labeled data)
def train_supervised_models(X_train, y_train, X_test, y_test):
    """Train supervised anomaly detection models"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    return results

# Train models for each dataset
print("Training models for Network Intrusion Detection...")
# Use only normal data for unsupervised models
X_train_normal_network = X_train_scaled_network[y_train_network == 0]
unsupervised_network = train_unsupervised_models(X_train_normal_network, X_test_scaled_network, y_test_network)
supervised_network = train_supervised_models(X_train_scaled_network, y_train_network, X_test_scaled_network, y_test_network)

print("\nTraining models for Credit Card Fraud Detection...")
X_train_normal_credit = X_train_scaled_credit[y_train_credit == 0]
unsupervised_credit = train_unsupervised_models(X_train_normal_credit, X_test_scaled_credit, y_test_credit)
supervised_credit = train_supervised_models(X_train_scaled_credit, y_train_credit, X_test_scaled_credit, y_test_credit)

print("\nTraining models for Manufacturing Quality Control...")
X_train_normal_manufacturing = X_train_scaled_manufacturing[y_train_manufacturing == 0]
unsupervised_manufacturing = train_unsupervised_models(X_train_normal_manufacturing, X_test_scaled_manufacturing, y_test_manufacturing)
supervised_manufacturing = train_supervised_models(X_train_scaled_manufacturing, y_train_manufacturing, X_test_scaled_manufacturing, y_test_manufacturing)

# 5. Time Series Anomaly Detection
print("\n=== Time Series Anomaly Detection ===")

def generate_time_series_data(n_points=2000, anomaly_ratio=0.05):
    """Generate synthetic time series data with anomalies"""
    np.random.seed(42)
    
    # Generate normal time series
    t = np.arange(n_points)
    normal_series = 10 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 0.5, n_points)
    
    # Add anomalies
    n_anomalies = int(n_points * anomaly_ratio)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    
    # Create different types of anomalies
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['spike', 'level_shift', 'trend_change'])
        
        if anomaly_type == 'spike':
            normal_series[idx] += np.random.normal(0, 5)
        elif anomaly_type == 'level_shift':
            normal_series[idx:] += np.random.normal(0, 3)
        else:  # trend_change
            normal_series[idx:] += np.cumsum(np.random.normal(0, 0.1, len(normal_series[idx:])))
    
    # Create labels
    labels = np.zeros(n_points)
    labels[anomaly_indices] = 1
    
    return pd.DataFrame({
        'timestamp': t,
        'value': normal_series,
        'anomaly': labels
    })

# Generate time series data
time_series_data = generate_time_series_data()
print(f"Time series data shape: {time_series_data.shape}")
print(f"Anomaly rate: {time_series_data['anomaly'].mean():.3f}")

# Simple time series anomaly detection using rolling statistics
def detect_time_series_anomalies(data, window=50, threshold=3):
    """Detect anomalies using rolling mean and standard deviation"""
    rolling_mean = data['value'].rolling(window=window, center=True).mean()
    rolling_std = data['value'].rolling(window=window, center=True).std()
    
    # Calculate z-scores
    z_scores = np.abs((data['value'] - rolling_mean) / rolling_std)
    
    # Detect anomalies
    anomalies = z_scores > threshold
    
    return anomalies.fillna(False)

# Apply time series anomaly detection
time_series_anomalies = detect_time_series_anomalies(time_series_data)
time_series_accuracy = accuracy_score(time_series_data['anomaly'], time_series_anomalies)
time_series_precision = precision_score(time_series_data['anomaly'], time_series_anomalies, zero_division=0)
time_series_recall = recall_score(time_series_data['anomaly'], time_series_anomalies, zero_division=0)

print(f"Time Series Anomaly Detection Results:")
print(f"Accuracy: {time_series_accuracy:.3f}")
print(f"Precision: {time_series_precision:.3f}")
print(f"Recall: {time_series_recall:.3f}")

# 6. Visualization and Comparison
print("\n=== Visualization and Comparison ===")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Function to plot results
def plot_anomaly_results(results, title, ax, metric='f1'):
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    bars = ax.bar(models, values, alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{title} - {metric.upper()}')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

# Plot F1 scores for supervised models
plot_anomaly_results(supervised_network, 'Network Intrusion Detection', axes[0, 0], 'f1')
plot_anomaly_results(supervised_credit, 'Credit Card Fraud Detection', axes[0, 1], 'f1')
plot_anomaly_results(supervised_manufacturing, 'Manufacturing Quality Control', axes[0, 2], 'f1')

# Plot ROC curves for supervised models
def plot_roc_curves(results, title, ax):
    for name, result in results.items():
        if 'probabilities' in result:
            fpr, tpr, _ = roc_curve(y_test_network, result['probabilities'])
            auc = result['auc']
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plot_roc_curves(supervised_network, 'Network Intrusion Detection', axes[1, 0])
plot_roc_curves(supervised_credit, 'Credit Card Fraud Detection', axes[1, 1])
plot_roc_curves(supervised_manufacturing, 'Manufacturing Quality Control', axes[1, 2])

plt.tight_layout()
plt.show()

# 7. Time Series Visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Plot time series with anomalies
axes[0].plot(time_series_data['timestamp'], time_series_data['value'], label='Time Series', alpha=0.7)
anomaly_points = time_series_data[time_series_data['anomaly'] == 1]
axes[0].scatter(anomaly_points['timestamp'], anomaly_points['value'], 
               color='red', s=50, label='Actual Anomalies', alpha=0.8)
axes[0].set_xlabel('Timestamp')
axes[0].set_ylabel('Value')
axes[0].set_title('Time Series with Anomalies')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot detected anomalies
axes[1].plot(time_series_data['timestamp'], time_series_data['value'], label='Time Series', alpha=0.7)
detected_anomalies = time_series_data[time_series_anomalies]
axes[1].scatter(detected_anomalies['timestamp'], detected_anomalies['value'], 
               color='orange', s=50, label='Detected Anomalies', alpha=0.8)
axes[1].set_xlabel('Timestamp')
axes[1].set_ylabel('Value')
axes[1].set_title('Time Series with Detected Anomalies')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# Get feature importance from Random Forest models
rf_network = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled_network, y_train_network)
rf_credit = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled_credit, y_train_credit)
rf_manufacturing = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled_manufacturing, y_train_manufacturing)

# Plot feature importance
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Network feature importance
network_importance = pd.DataFrame({
    'feature': network_features,
    'importance': rf_network.feature_importances_
}).sort_values('importance', ascending=True)

axes[0].barh(network_importance['feature'], network_importance['importance'])
axes[0].set_title('Feature Importance - Network Intrusion Detection')
axes[0].set_xlabel('Importance')

# Credit feature importance
credit_importance = pd.DataFrame({
    'feature': credit_features,
    'importance': rf_credit.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(credit_importance['feature'], credit_importance['importance'])
axes[1].set_title('Feature Importance - Credit Card Fraud Detection')
axes[1].set_xlabel('Importance')

# Manufacturing feature importance
manufacturing_importance = pd.DataFrame({
    'feature': manufacturing_features,
    'importance': rf_manufacturing.feature_importances_
}).sort_values('importance', ascending=True)

axes[2].barh(manufacturing_importance['feature'], manufacturing_importance['importance'])
axes[2].set_title('Feature Importance - Manufacturing Quality Control')
axes[2].set_xlabel('Importance')

plt.tight_layout()
plt.show()

# 9. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

# Find best model for each problem
def find_best_model(results, metric='f1'):
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1]

print("Best Models by F1 Score:")
print(f"1. Network Intrusion Detection:")
best_network, network_metrics = find_best_model(supervised_network)
print(f"   Best model: {best_network}")
print(f"   F1: {network_metrics['f1']:.3f}")
print(f"   AUC: {network_metrics['auc']:.3f}")

print(f"\n2. Credit Card Fraud Detection:")
best_credit, credit_metrics = find_best_model(supervised_credit)
print(f"   Best model: {best_credit}")
print(f"   F1: {credit_metrics['f1']:.3f}")
print(f"   AUC: {credit_metrics['auc']:.3f}")

print(f"\n3. Manufacturing Quality Control:")
best_manufacturing, manufacturing_metrics = find_best_model(supervised_manufacturing)
print(f"   Best model: {best_manufacturing}")
print(f"   F1: {manufacturing_metrics['f1']:.3f}")
print(f"   AUC: {manufacturing_metrics['auc']:.3f}")

print(f"\n4. Time Series Anomaly Detection:")
print(f"   Method: Rolling Statistics")
print(f"   F1: {f1_score(time_series_data['anomaly'], time_series_anomalies, zero_division=0):.3f}")

print(f"\nKey Insights:")
print(f"- Supervised models generally outperform unsupervised models")
print(f"- Random Forest shows good performance across all domains")
print(f"- Feature engineering is crucial for anomaly detection")
print(f"- Time series anomalies require specialized approaches")

print(f"\nRecommendations:")
print(f"- Use supervised models when labeled data is available")
print(f"- Use unsupervised models for unknown anomaly types")
print(f"- Combine multiple approaches for better performance")
print(f"- Consider domain-specific features and constraints")
print(f"- Regularly retrain models as patterns change")
print(f"- Use ensemble methods for robust detection") 