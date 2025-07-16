"""
Deep Neural Networks
====================

This script demonstrates Deep Neural Networks (DNN) including:
- Basic DNN architecture
- Different activation functions
- Regularization techniques
- Hyperparameter tuning
- DNN for different tasks (classification, regression)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression, load_breast_cancer, load_diabetes
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_synthetic_classification_data(n_samples=1000, n_features=20, n_classes=3):
    """Create synthetic classification data"""
    print("Creating synthetic classification data...")
    
    # Ensure n_features is large enough
    n_informative = min(15, n_features // 2)
    n_redundant = min(5, n_features // 4)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print(f"Classification data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

def create_synthetic_regression_data(n_samples=1000, n_features=20):
    """Create synthetic regression data"""
    print("Creating synthetic regression data...")
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_targets=1,
        noise=0.1,
        random_state=42
    )
    
    print(f"Regression data shape: X={X.shape}, y={y.shape}")
    return X, y

def create_complex_nonlinear_data(n_samples=1000):
    """Create complex nonlinear data for DNN demonstration"""
    print("Creating complex nonlinear data...")
    
    # Generate input features
    X = np.random.randn(n_samples, 10)
    
    # Create complex nonlinear target
    y = (np.sin(X[:, 0]) * np.cos(X[:, 1]) + 
         X[:, 2]**2 + 
         np.exp(X[:, 3]) + 
         X[:, 4] * X[:, 5] + 
         np.random.normal(0, 0.1, n_samples))
    
    print(f"Nonlinear data shape: X={X.shape}, y={y.shape}")
    return X, y

def build_basic_dnn(input_dim, num_classes=1, task='regression'):
    """Build a basic DNN architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_deep_dnn(input_dim, num_classes=1, task='regression'):
    """Build a deeper DNN architecture"""
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_regularized_dnn(input_dim, num_classes=1, task='regression'):
    """Build DNN with regularization techniques"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, 
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_dnn_with_different_activations(input_dim, num_classes=1, task='regression'):
    """Build DNN with different activation functions"""
    model = Sequential([
        Dense(128, input_dim=input_dim),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        Dense(64),
        BatchNormalization(),
        Activation('tanh'),
        Dropout(0.3),
        
        Dense(32),
        BatchNormalization(),
        Activation('swish'),  # Swish activation
        Dropout(0.3),
        
        Dense(num_classes, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title(f'{model_name} - Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy/MAE plot
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title(f'{model_name} - Training and Validation Accuracy')
        axes[1].set_ylabel('Accuracy')
    else:
        axes[1].plot(history.history['mae'], label='Training MAE')
        axes[1].plot(history.history['val_mae'], label='Validation MAE')
        axes[1].set_title(f'{model_name} - Training and Validation MAE')
        axes[1].set_ylabel('MAE')
    
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_regression_results(y_true, y_pred, model_name):
    """Plot regression results"""
    plt.figure(figsize=(12, 4))
    
    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.grid(True)
    
    # Residuals
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residuals')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def classification_example():
    """Example: DNN for classification"""
    print("="*60)
    print("DNN CLASSIFICATION EXAMPLE")
    print("="*60)
    
    # Load breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Build and train DNN
    model = build_basic_dnn(X_train_scaled.shape[1], 2, 'classification')
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot results
    plot_training_history(history, "DNN Classification")
    plot_confusion_matrix(y_test, y_pred_classes, ['Benign', 'Malignant'])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['Benign', 'Malignant']))
    
    return model, history

def regression_example():
    """Example: DNN for regression"""
    print("\n" + "="*60)
    print("DNN REGRESSION EXAMPLE")
    print("="*60)
    
    # Load diabetes dataset
    data = load_diabetes()
    X, y = data.data, data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Build and train DNN
    model = build_basic_dnn(X_train_scaled.shape[1], 1, 'regression')
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled).flatten()
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plot_training_history(history, "DNN Regression")
    plot_regression_results(y_test, y_pred, "DNN Regression")
    
    return model, history

def nonlinear_regression_example():
    """Example: DNN for complex nonlinear regression"""
    print("\n" + "="*60)
    print("DNN NONLINEAR REGRESSION EXAMPLE")
    print("="*60)
    
    # Create complex nonlinear data
    X, y = create_complex_nonlinear_data(1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Build deep DNN for complex patterns
    model = build_deep_dnn(X_train_scaled.shape[1], 1, 'regression')
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled).flatten()
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot results
    plot_training_history(history, "Deep DNN Nonlinear Regression")
    plot_regression_results(y_test, y_pred, "Deep DNN Nonlinear Regression")
    
    return model, history

def regularization_comparison():
    """Compare different regularization techniques"""
    print("\n" + "="*60)
    print("REGULARIZATION COMPARISON")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_classification_data(1000, 50, 3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    
    # Build different models
    models = {
        'Basic DNN': build_basic_dnn(X_train_scaled.shape[1], 3, 'classification'),
        'Regularized DNN': build_regularized_dnn(X_train_scaled.shape[1], 3, 'classification')
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
        
        results[name] = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'history': history
        }
        
        print(f"{name} - Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history'].history['val_accuracy'], label=name)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['history'].history['val_loss'], label=name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    for name, result in results.items():
        print(f"{name}: Accuracy = {result['accuracy']:.4f}, Loss = {result['loss']:.4f}")

def activation_functions_comparison():
    """Compare different activation functions"""
    print("\n" + "="*60)
    print("ACTIVATION FUNCTIONS COMPARISON")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_regression_data(1000, 20)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build models with different activations
    models = {
        'ReLU': build_basic_dnn(X_train_scaled.shape[1], 1, 'regression'),
        'Mixed Activations': build_dnn_with_different_activations(X_train_scaled.shape[1], 1, 'regression')
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
        
        results[name] = {
            'mae': test_mae,
            'loss': test_loss,
            'history': history
        }
        
        print(f"{name} - MAE: {test_mae:.4f}, Loss: {test_loss:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history'].history['val_mae'], label=name)
    plt.title('Validation MAE Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['history'].history['val_loss'], label=name)
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    for name, result in results.items():
        print(f"{name}: MAE = {result['mae']:.4f}, Loss = {result['loss']:.4f}")

def hyperparameter_tuning_example():
    """Example: Hyperparameter tuning for DNN"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING EXAMPLE")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_classification_data(800, 30, 2)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [16, 32, 64],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    print("Parameter grid for tuning:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Simple grid search (in practice, use Keras Tuner or Optuna)
    best_score = 0
    best_params = None
    
    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for dropout_rate in param_grid['dropout_rate']:
                print(f"\nTesting: lr={lr}, batch_size={batch_size}, dropout={dropout_rate}")
                
                # Build model with current parameters
                model = Sequential([
                    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
                    Dropout(dropout_rate),
                    Dense(32, activation='relu'),
                    Dropout(dropout_rate),
                    Dense(2, activation='softmax')
                ])
                
                model.compile(optimizer=Adam(learning_rate=lr),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
                
                # Train model
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(
                    X_train_scaled, y_train_cat,
                    epochs=30,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Evaluate model
                test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_cat, verbose=0)
                
                if test_accuracy > best_score:
                    best_score = test_accuracy
                    best_params = {'learning_rate': lr, 'batch_size': batch_size, 'dropout_rate': dropout_rate}
                
                print(f"  Test Accuracy: {test_accuracy:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")

def main():
    """Main function to run all DNN examples"""
    print("=== DEEP NEURAL NETWORKS EXAMPLES ===\n")
    
    # 1. Classification example
    classification_example()
    
    # 2. Regression example
    regression_example()
    
    # 3. Nonlinear regression example
    nonlinear_regression_example()
    
    # 4. Regularization comparison
    regularization_comparison()
    
    # 5. Activation functions comparison
    activation_functions_comparison()
    
    # 6. Hyperparameter tuning
    hyperparameter_tuning_example()
    
    print("\n" + "="*60)
    print("=== DNN EXAMPLES COMPLETE! ===")
    print("="*60)

if __name__ == "__main__":
    main() 