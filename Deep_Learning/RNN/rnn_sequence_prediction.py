"""
RNN Sequence Prediction
=======================

This script demonstrates Recurrent Neural Networks (RNN) for sequence prediction including:
- Simple RNN implementation
- LSTM implementation
- GRU implementation
- Time series prediction
- Text sequence prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_sine_wave_data(n_samples=1000, sequence_length=50):
    """Create sine wave time series data"""
    print("Creating sine wave time series data...")
    
    # Generate time points
    t = np.linspace(0, 100, n_samples)
    
    # Create sine wave with noise
    sine_wave = np.sin(0.1 * t) + 0.1 * np.random.randn(n_samples)
    
    # Create sequences
    X, y = [], []
    for i in range(len(sine_wave) - sequence_length):
        X.append(sine_wave[i:(i + sequence_length)])
        y.append(sine_wave[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for RNN input: (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"Sine wave data shape: X={X.shape}, y={y.shape}")
    return X, y

def create_stock_price_data(n_samples=1000, sequence_length=30):
    """Create simulated stock price data"""
    print("Creating simulated stock price data...")
    
    # Generate random walk for stock prices
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = [100]  # Starting price
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(len(prices) - sequence_length):
        X.append(prices[i:(i + sequence_length)])
        y.append(prices[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    
    print(f"Stock price data shape: X={X_scaled.shape}, y={y_scaled.shape}")
    return X_scaled, y_scaled, scaler

def create_text_sequence_data(text_length=10000, sequence_length=10):
    """Create text sequence data for character-level prediction"""
    print("Creating text sequence data...")
    
    # Simple text corpus
    text = """
    The quick brown fox jumps over the lazy dog. 
    Machine learning is a subset of artificial intelligence. 
    Deep learning uses neural networks with multiple layers.
    Recurrent neural networks are good for sequence data.
    """ * 100
    
    # Create character mapping
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create sequences
    X, y = [], []
    for i in range(len(text) - sequence_length):
        seq = text[i:i + sequence_length]
        target = text[i + sequence_length]
        X.append([char_to_int[ch] for ch in seq])
        y.append(char_to_int[target])
    
    X = np.array(X)
    y = np.array(y)
    
    # One-hot encode
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=len(chars))
    
    print(f"Text sequence data shape: X={X.shape}, y={y_onehot.shape}")
    print(f"Vocabulary size: {len(chars)}")
    
    return X, y_onehot, char_to_int, int_to_char, chars

def build_simple_rnn_model(input_shape, output_size=1):
    """Build a simple RNN model"""
    model = Sequential([
        SimpleRNN(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(output_size)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_lstm_model(input_shape, output_size=1):
    """Build an LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(output_size)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_gru_model(input_shape, output_size=1):
    """Build a GRU model"""
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(output_size)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_text_rnn_model(vocab_size, sequence_length, output_size):
    """Build RNN model for text generation"""
    model = Sequential([
        SimpleRNN(128, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        SimpleRNN(128, return_sequences=False),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(output_size, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, epochs=50):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name} Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    return history, y_pred, mse, mae

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
    
    # If accuracy is available (for classification tasks)
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title(f'{model_name} - Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(y_true, y_pred, model_name, scaler=None):
    """Plot predictions vs actual values"""
    if scaler:
        # Inverse transform if scaler is provided
        y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_original = y_true
        y_pred_original = y_pred
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.subplot(1, 2, 1)
    plt.plot(y_true_original, label='Actual', alpha=0.7)
    plt.plot(y_pred_original, label='Predicted', alpha=0.7)
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true_original, y_pred_original, alpha=0.6)
    plt.plot([y_true_original.min(), y_true_original.max()], 
             [y_true_original.min(), y_true_original.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Scatter Plot')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def generate_text(model, char_to_int, int_to_char, seed_text, length=100):
    """Generate text using trained model"""
    generated_text = seed_text
    
    for _ in range(length):
        # Prepare input sequence
        x = np.array([char_to_int[ch] for ch in seed_text])
        x = x.reshape(1, len(x), 1)
        
        # Predict next character
        pred = model.predict(x, verbose=0)
        pred_char_idx = np.argmax(pred[0])
        pred_char = int_to_char[pred_char_idx]
        
        # Add to generated text
        generated_text += pred_char
        seed_text = seed_text[1:] + pred_char
    
    return generated_text

def sine_wave_prediction_example():
    """Example: Sine wave prediction using different RNN types"""
    print("="*60)
    print("SINE WAVE PREDICTION EXAMPLE")
    print("="*60)
    
    # Create data
    X, y = create_sine_wave_data(1000, 50)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train models
    models = {
        'Simple RNN': build_simple_rnn_model((X.shape[1], X.shape[2])),
        'LSTM': build_lstm_model((X.shape[1], X.shape[2])),
        'GRU': build_gru_model((X.shape[1], X.shape[2]))
    }
    
    results = {}
    
    for name, model in models.items():
        history, y_pred, mse, mae = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, name, epochs=30
        )
        
        results[name] = {
            'history': history,
            'predictions': y_pred,
            'mse': mse,
            'mae': mae
        }
        
        # Plot training history
        plot_training_history(history, name)
        
        # Plot predictions
        plot_predictions(y_test, y_pred, name)
    
    # Compare models
    print("\nModel Comparison:")
    for name, result in results.items():
        print(f"{name}: MSE = {result['mse']:.6f}, MAE = {result['mae']:.6f}")

def stock_price_prediction_example():
    """Example: Stock price prediction"""
    print("\n" + "="*60)
    print("STOCK PRICE PREDICTION EXAMPLE")
    print("="*60)
    
    # Create data
    X, y, scaler = create_stock_price_data(1000, 30)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train LSTM model (best for financial data)
    model = build_lstm_model((X.shape[1], X.shape[2]))
    
    history, y_pred, mse, mae = train_and_evaluate_model(
        model, X_train, y_train, X_test, y_test, "LSTM Stock Predictor", epochs=50
    )
    
    # Plot results
    plot_training_history(history, "LSTM Stock Predictor")
    plot_predictions(y_test, y_pred, "LSTM Stock Predictor", scaler)
    
    print(f"\nStock Price Prediction Results:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")

def text_generation_example():
    """Example: Text generation using RNN"""
    print("\n" + "="*60)
    print("TEXT GENERATION EXAMPLE")
    print("="*60)
    
    # Create data
    X, y, char_to_int, int_to_char, chars = create_text_sequence_data(5000, 20)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train model
    model = build_text_rnn_model(len(chars), X.shape[1], len(chars))
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history, "Text Generation RNN")
    
    # Generate text
    seed_text = "The quick brown fox"
    generated_text = generate_text(model, char_to_int, int_to_char, seed_text, 100)
    
    print(f"\nGenerated Text:")
    print(f"Seed: {seed_text}")
    print(f"Generated: {generated_text}")

def main():
    """Main function to run all RNN examples"""
    print("=== RNN SEQUENCE PREDICTION EXAMPLES ===\n")
    
    # 1. Sine wave prediction
    sine_wave_prediction_example()
    
    # 2. Stock price prediction
    stock_price_prediction_example()
    
    # 3. Text generation
    text_generation_example()
    
    print("\n" + "="*60)
    print("=== RNN EXAMPLES COMPLETE! ===")
    print("="*60)

if __name__ == "__main__":
    main() 