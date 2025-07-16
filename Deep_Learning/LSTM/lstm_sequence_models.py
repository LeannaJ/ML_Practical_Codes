"""
LSTM Sequence Models
====================

This script demonstrates Long Short-Term Memory (LSTM) networks including:
- Basic LSTM implementation
- LSTM for time series prediction
- LSTM for text classification
- LSTM for text generation
- Bidirectional LSTM
- LSTM with attention
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional, Input
from tensorflow.keras.layers import GlobalAveragePooling1D, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import re
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_time_series_data(n_samples=1000, seq_length=50):
    """Create synthetic time series data"""
    print("Creating synthetic time series data...")
    
    # Generate time points
    t = np.linspace(0, 100, n_samples)
    
    # Create multiple time series with different patterns
    series1 = np.sin(0.1 * t) + 0.1 * np.random.randn(n_samples)  # Sine wave
    series2 = np.cos(0.15 * t) + 0.1 * np.random.randn(n_samples)  # Cosine wave
    series3 = 0.5 * np.sin(0.05 * t) + 0.3 * np.cos(0.2 * t) + 0.1 * np.random.randn(n_samples)  # Combined
    
    # Combine into multivariate time series
    data = np.column_stack([series1, series2, series3])
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 0])  # Predict first series
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Time series data shape: X={X.shape}, y={y.shape}")
    return X, y

def create_stock_price_data(n_samples=1000, seq_length=30):
    """Create simulated stock price data"""
    print("Creating simulated stock price data...")
    
    # Generate multiple stock prices with correlations
    np.random.seed(42)
    
    # Stock 1: Random walk
    returns1 = np.random.normal(0.001, 0.02, n_samples)
    prices1 = [100]
    for ret in returns1:
        prices1.append(prices1[-1] * (1 + ret))
    
    # Stock 2: Correlated with stock 1
    returns2 = 0.7 * returns1 + 0.3 * np.random.normal(0.001, 0.02, n_samples)
    prices2 = [100]
    for ret in returns2:
        prices2.append(prices2[-1] * (1 + ret))
    
    # Stock 3: Different pattern
    returns3 = np.random.normal(0.0005, 0.015, n_samples)
    prices3 = [100]
    for ret in returns3:
        prices3.append(prices3[-1] * (1 + ret))
    
    # Combine prices
    prices = np.column_stack([prices1, prices2, prices3])
    
    # Create sequences
    X, y = [], []
    for i in range(len(prices) - seq_length):
        X.append(prices[i:(i + seq_length)])
        y.append(prices[i + seq_length, 0])  # Predict stock 1
    
    X = np.array(X)
    y = np.array(y)
    
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()
    
    print(f"Stock price data shape: X={X_scaled.shape}, y={y_scaled.shape}")
    return X_scaled, y_scaled, scaler

def create_text_classification_data():
    """Create synthetic text classification data"""
    print("Creating synthetic text classification data...")
    
    # Simple text corpus for sentiment analysis
    positive_texts = [
        "I love this product, it's amazing!",
        "Great service and excellent quality",
        "Highly recommended, very satisfied",
        "Outstanding performance and reliability",
        "Fantastic experience, would buy again",
        "Excellent customer support",
        "Perfect for my needs",
        "Wonderful product, exceeded expectations",
        "Best purchase I've ever made",
        "Absolutely love it, great value"
    ] * 50
    
    negative_texts = [
        "Terrible product, waste of money",
        "Poor quality and bad service",
        "Disappointed with this purchase",
        "Not worth the price at all",
        "Awful experience, would not recommend",
        "Bad customer service",
        "Defective product, very frustrated",
        "Worst purchase ever",
        "Complete waste of time and money",
        "Avoid this product at all costs"
    ] * 50
    
    neutral_texts = [
        "The product is okay, nothing special",
        "Average quality, meets basic needs",
        "It works as expected",
        "Standard product, neither good nor bad",
        "Acceptable for the price",
        "Moderate performance",
        "Fair quality, could be better",
        "Decent product, nothing remarkable",
        "Satisfactory but not impressive",
        "It's fine, nothing to complain about"
    ] * 50
    
    texts = positive_texts + negative_texts + neutral_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts) + [2] * len(neutral_texts)
    
    print(f"Text classification data: {len(texts)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    return texts, labels

def create_text_generation_data():
    """Create text generation data"""
    print("Creating text generation data...")
    
    # Simple text corpus for character-level generation
    text = """
    The quick brown fox jumps over the lazy dog. 
    Machine learning is a subset of artificial intelligence. 
    Deep learning uses neural networks with multiple layers.
    Long short-term memory networks are good for sequence data.
    Natural language processing helps computers understand human language.
    Computer vision enables machines to interpret visual information.
    Reinforcement learning teaches agents to make decisions.
    Supervised learning uses labeled training data.
    Unsupervised learning finds patterns in unlabeled data.
    """ * 20
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    print(f"Text generation corpus length: {len(text)} characters")
    return text

def build_basic_lstm_model(input_shape, output_size=1, task='regression'):
    """Build a basic LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_deep_lstm_model(input_shape, output_size=1, task='regression'):
    """Build a deeper LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_bidirectional_lstm_model(input_shape, output_size=1, task='regression'):
    """Build a bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(output_size, activation='linear' if task == 'regression' else 'softmax')
    ])
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_lstm_with_attention_model(input_shape, output_size=1, task='regression'):
    """Build LSTM model with attention mechanism"""
    inputs = Input(shape=input_shape)
    
    # LSTM layers
    lstm_out = LSTM(50, return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    lstm_out = LSTM(50, return_sequences=True)(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Attention mechanism
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=50)(lstm_out, lstm_out)
    attention_out = Concatenate()([lstm_out, attention])
    
    # Global pooling
    pooled = GlobalAveragePooling1D()(attention_out)
    
    # Dense layers
    dense_out = Dense(32, activation='relu')(pooled)
    dense_out = Dropout(0.3)(dense_out)
    outputs = Dense(output_size, activation='linear' if task == 'regression' else 'softmax')(dense_out)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    if task == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return model

def build_text_lstm_model(vocab_size, max_length, num_classes):
    """Build LSTM model for text classification"""
    model = Sequential([
        Embedding(vocab_size, 100, input_length=max_length),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def build_text_generation_lstm_model(vocab_size, max_length):
    """Build LSTM model for text generation"""
    model = Sequential([
        Embedding(vocab_size, 128, input_length=max_length),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(vocab_size, activation='softmax')
    ])
    
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

def plot_time_series_predictions(y_true, y_pred, model_name, scaler=None):
    """Plot time series predictions"""
    if scaler:
        # Inverse transform if scaler is provided
        y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_original = y_true
        y_pred_original = y_pred
    
    plt.figure(figsize=(12, 6))
    
    # Plot actual vs predicted
    plt.plot(y_true_original, label='Actual', alpha=0.7)
    plt.plot(y_pred_original, label='Predicted', alpha=0.7)
    plt.title(f'{model_name} - Time Series Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
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

def generate_text(model, tokenizer, seed_text, max_length, num_chars=100):
    """Generate text using trained LSTM model"""
    generated_text = seed_text
    
    for _ in range(num_chars):
        # Tokenize the seed text
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        
        # Predict next character
        predicted = model.predict(token_list, verbose=0)
        predicted_token = np.argmax(predicted[0])
        
        # Convert token back to character
        for word, index in tokenizer.word_index.items():
            if index == predicted_token:
                output_word = word
                break
        else:
            output_word = ""
        
        generated_text += " " + output_word
    
    return generated_text

def time_series_prediction_example():
    """Example: LSTM for time series prediction"""
    print("="*60)
    print("LSTM TIME SERIES PREDICTION")
    print("="*60)
    
    # Create time series data
    X, y = create_time_series_data(1000, 50)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train LSTM model
    model = build_basic_lstm_model((X.shape[1], X.shape[2]))
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    
    # Plot results
    plot_training_history(history, "LSTM Time Series")
    plot_time_series_predictions(y_test, y_pred, "LSTM Time Series")
    
    return model, history

def stock_price_prediction_example():
    """Example: LSTM for stock price prediction"""
    print("\n" + "="*60)
    print("LSTM STOCK PRICE PREDICTION")
    print("="*60)
    
    # Create stock price data
    X, y, scaler = create_stock_price_data(1000, 30)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train deep LSTM model
    model = build_deep_lstm_model((X.shape[1], X.shape[2]))
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}")
    
    # Plot results
    plot_training_history(history, "Deep LSTM Stock Prediction")
    plot_time_series_predictions(y_test, y_pred, "Deep LSTM Stock Prediction", scaler)
    
    return model, history

def text_classification_example():
    """Example: LSTM for text classification"""
    print("\n" + "="*60)
    print("LSTM TEXT CLASSIFICATION")
    print("="*60)
    
    # Create text classification data
    texts, labels = create_text_classification_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    
    # Tokenize text
    max_words = 1000
    max_length = 50
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    
    print(f"Training set: {X_train_padded.shape}")
    print(f"Test set: {X_test_padded.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    
    # Build and train LSTM model
    model = build_text_lstm_model(max_words, max_length, 3)
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train_padded, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot results
    plot_training_history(history, "LSTM Text Classification")
    plot_confusion_matrix(y_test, y_pred_classes, ['Negative', 'Positive', 'Neutral'])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['Negative', 'Positive', 'Neutral']))
    
    return model, history

def text_generation_example():
    """Example: LSTM for text generation"""
    print("\n" + "="*60)
    print("LSTM TEXT GENERATION")
    print("="*60)
    
    # Create text generation data
    text = create_text_generation_data()
    
    # Tokenize text
    max_words = 100
    max_length = 20
    tokenizer = Tokenizer(num_words=max_words, char_level=True)
    tokenizer.fit_on_texts([text])
    
    # Create sequences
    sequences = []
    for i in range(len(text) - max_length):
        sequences.append(text[i:i + max_length + 1])
    
    # Convert to sequences
    X = []
    y = []
    for seq in sequences:
        X.append(seq[:-1])
        y.append(seq[-1])
    
    # Convert to numerical sequences
    X_seq = tokenizer.texts_to_sequences(X)
    y_seq = tokenizer.texts_to_sequences([y])[0]
    
    # Pad sequences
    X_padded = pad_sequences(X_seq, maxlen=max_length, padding='pre')
    
    # Convert y to categorical
    y_cat = to_categorical(y_seq, num_classes=max_words)
    
    # Split data
    split_idx = int(0.8 * len(X_padded))
    X_train, X_test = X_padded[:split_idx], X_padded[split_idx:]
    y_train, y_test = y_cat[:split_idx], y_cat[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Vocabulary size: {max_words}")
    
    # Build and train LSTM model
    model = build_text_generation_lstm_model(max_words, max_length)
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train model
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
    
    # Generate text
    seed_text = "The quick brown fox"
    generated_text = generate_text(model, tokenizer, seed_text, max_length, 50)
    
    print(f"\nGenerated Text:")
    print(f"Seed: {seed_text}")
    print(f"Generated: {generated_text}")
    
    # Plot results
    plot_training_history(history, "LSTM Text Generation")
    
    return model, history

def lstm_architecture_comparison():
    """Compare different LSTM architectures"""
    print("\n" + "="*60)
    print("LSTM ARCHITECTURE COMPARISON")
    print("="*60)
    
    # Create time series data
    X, y = create_time_series_data(800, 30)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build different models
    models = {
        'Basic LSTM': build_basic_lstm_model((X.shape[1], X.shape[2])),
        'Deep LSTM': build_deep_lstm_model((X.shape[1], X.shape[2])),
        'Bidirectional LSTM': build_bidirectional_lstm_model((X.shape[1], X.shape[2])),
        'LSTM with Attention': build_lstm_with_attention_model((X.shape[1], X.shape[2]))
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
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

def main():
    """Main function to run all LSTM examples"""
    print("=== LSTM SEQUENCE MODELS EXAMPLES ===\n")
    
    # 1. Time series prediction
    time_series_prediction_example()
    
    # 2. Stock price prediction
    stock_price_prediction_example()
    
    # 3. Text classification
    text_classification_example()
    
    # 4. Text generation
    text_generation_example()
    
    # 5. Architecture comparison
    lstm_architecture_comparison()
    
    print("\n" + "="*60)
    print("=== LSTM EXAMPLES COMPLETE! ===")
    print("="*60)

if __name__ == "__main__":
    main() 