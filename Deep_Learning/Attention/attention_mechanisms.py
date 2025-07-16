"""
Attention Mechanisms
===================

This script demonstrates various attention mechanisms including:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention
- Attention visualization
- Sequence-to-sequence with attention
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Scaled Dot-Product Attention implementation"""
    
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def call(self, Q, K, V, mask=None):
        # Calculate attention scores
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        
        # Apply mask if provided
        if mask is not None:
            scores += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention weights to values
        output = tf.matmul(attention_weights, V)
        
        return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention implementation"""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.W_q = Dense(d_model)
        self.W_k = Dense(d_model)
        self.W_v = Dense(d_model)
        self.W_o = Dense(d_model)
        
        # Attention layer
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def call(self, Q, K, V, mask=None):
        batch_size = tf.shape(Q)[0]
        
        # Linear transformations
        Q = self.W_q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_k(K)  # (batch_size, seq_len, d_model)
        V = self.W_v(V)  # (batch_size, seq_len, d_model)
        
        # Split into multiple heads
        Q = tf.reshape(Q, [batch_size, -1, self.num_heads, self.d_k])
        K = tf.reshape(K, [batch_size, -1, self.num_heads, self.d_k])
        V = tf.reshape(V, [batch_size, -1, self.num_heads, self.d_k])
        
        # Transpose for attention computation
        Q = tf.transpose(Q, [0, 2, 1, 3])  # (batch_size, num_heads, seq_len, d_k)
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        # Apply attention
        output, attention_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        output = tf.transpose(output, [0, 2, 1, 3])  # (batch_size, seq_len, num_heads, d_k)
        output = tf.reshape(output, [batch_size, -1, self.d_model])  # (batch_size, seq_len, d_model)
        
        # Final linear transformation
        output = self.W_o(output)
        
        return output, attention_weights

class SelfAttention(tf.keras.layers.Layer):
    """Self-Attention layer"""
    
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
    
    def call(self, x, mask=None):
        # Apply multi-head attention
        attn_output, attention_weights = self.multi_head_attention(x, x, x, mask)
        
        # Add residual connection and layer normalization
        output = self.layer_norm(x + attn_output)
        
        return output, attention_weights

def create_sample_sequence_data(n_samples=1000, seq_length=20, vocab_size=100):
    """Create sample sequence data for attention demonstration"""
    print("Creating sample sequence data...")
    
    # Generate random sequences
    X = np.random.randint(1, vocab_size, size=(n_samples, seq_length))
    y = np.random.randint(0, 2, size=(n_samples,))  # Binary classification
    
    print(f"Sequence data shape: X={X.shape}, y={y.shape}")
    return X, y

def create_attention_demo_data():
    """Create simple data for attention visualization"""
    print("Creating attention demonstration data...")
    
    # Simple sequence: [1, 2, 3, 4, 5]
    sequence = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
    
    # Create embeddings (simple one-hot encoding for demo)
    embeddings = np.eye(6)[sequence.astype(int)]  # 6 for vocab size 0-5
    
    print(f"Demo sequence: {sequence[0]}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings

def build_attention_model(vocab_size, seq_length, d_model=64, num_heads=8):
    """Build a model with attention mechanism"""
    inputs = Input(shape=(seq_length,))
    
    # Embedding layer
    embedding = Embedding(vocab_size, d_model, input_length=seq_length)(inputs)
    
    # Self-attention layer
    attention_output, attention_weights = SelfAttention(d_model, num_heads)(embedding)
    
    # Global average pooling
    pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_output)
    
    # Dense layers for classification
    dense1 = Dense(64, activation='relu')(pooled)
    dropout = Dropout(0.5)(dense1)
    outputs = Dense(1, activation='sigmoid')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    
    return model, attention_weights

def visualize_attention_weights(attention_weights, sequence, title="Attention Weights"):
    """Visualize attention weights"""
    print(f"\nVisualizing {title}...")
    
    # Get attention weights for the first sample and first head
    attn_weights = attention_weights[0, 0].numpy()  # (seq_len, seq_len)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, 
                xticklabels=sequence, 
                yticklabels=sequence,
                cmap='Blues', 
                annot=True, 
                fmt='.2f',
                cbar_kws={'label': 'Attention Weight'})
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.show()
    
    # Print attention weights
    print(f"Attention weights shape: {attn_weights.shape}")
    print("Attention weights matrix:")
    print(attn_weights)

def demonstrate_scaled_dot_product_attention():
    """Demonstrate scaled dot-product attention"""
    print("="*60)
    print("SCALED DOT-PRODUCT ATTENTION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    embeddings = create_attention_demo_data()
    
    # Create Q, K, V matrices
    d_k = 4
    Q = tf.Variable(np.random.randn(1, 5, d_k), dtype=tf.float32)
    K = tf.Variable(np.random.randn(1, 5, d_k), dtype=tf.float32)
    V = tf.Variable(np.random.randn(1, 5, d_k), dtype=tf.float32)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Apply scaled dot-product attention
    attention_layer = ScaledDotProductAttention(d_k)
    output, attention_weights = attention_layer(Q, K, V)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights
    sequence = [1, 2, 3, 4, 5]
    visualize_attention_weights(attention_weights, sequence, "Scaled Dot-Product Attention")

def demonstrate_multi_head_attention():
    """Demonstrate multi-head attention"""
    print("\n" + "="*60)
    print("MULTI-HEAD ATTENTION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    embeddings = create_attention_demo_data()
    
    # Create multi-head attention layer
    d_model = 8
    num_heads = 2
    multi_head_attn = MultiHeadAttention(d_model, num_heads)
    
    # Apply multi-head attention
    output, attention_weights = multi_head_attn(embeddings, embeddings, embeddings)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights for first head
    sequence = [1, 2, 3, 4, 5]
    visualize_attention_weights(attention_weights, sequence, "Multi-Head Attention (Head 1)")

def demonstrate_self_attention():
    """Demonstrate self-attention"""
    print("\n" + "="*60)
    print("SELF-ATTENTION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    embeddings = create_attention_demo_data()
    
    # Create self-attention layer
    d_model = 8
    num_heads = 2
    self_attn = SelfAttention(d_model, num_heads)
    
    # Apply self-attention
    output, attention_weights = self_attn(embeddings)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Visualize attention weights
    sequence = [1, 2, 3, 4, 5]
    visualize_attention_weights(attention_weights, sequence, "Self-Attention")

def train_attention_model():
    """Train a model with attention mechanism"""
    print("\n" + "="*60)
    print("TRAINING MODEL WITH ATTENTION")
    print("="*60)
    
    # Create data
    vocab_size = 100
    seq_length = 20
    X, y = create_sample_sequence_data(1000, seq_length, vocab_size)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    d_model = 64
    num_heads = 8
    model, attention_weights = build_attention_model(vocab_size, seq_length, d_model, num_heads)
    
    print(f"Model summary:")
    model.summary()
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
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
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, history

def attention_analysis_example():
    """Analyze attention patterns in trained model"""
    print("\n" + "="*60)
    print("ATTENTION ANALYSIS EXAMPLE")
    print("="*60)
    
    # Create a simple test sequence
    test_sequence = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    
    # Build a simple attention model for analysis
    vocab_size = 11  # 0-10
    seq_length = 10
    d_model = 8
    num_heads = 2
    
    inputs = Input(shape=(seq_length,))
    embedding = Embedding(vocab_size, d_model, input_length=seq_length)(inputs)
    attention_output, attention_weights = SelfAttention(d_model, num_heads)(embedding)
    
    analysis_model = Model(inputs=inputs, outputs=[attention_output, attention_weights])
    
    # Get attention weights for test sequence
    _, attn_weights = analysis_model.predict(test_sequence)
    
    print(f"Test sequence: {test_sequence[0]}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Visualize attention weights
    sequence = test_sequence[0]
    visualize_attention_weights(attn_weights, sequence, "Attention Analysis")

def compare_attention_with_lstm():
    """Compare attention mechanism with LSTM"""
    print("\n" + "="*60)
    print("ATTENTION vs LSTM COMPARISON")
    print("="*60)
    
    # Create data
    vocab_size = 100
    seq_length = 20
    X, y = create_sample_sequence_data(1000, seq_length, vocab_size)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build Attention model
    attention_model, _ = build_attention_model(vocab_size, seq_length)
    
    # Build LSTM model
    lstm_inputs = Input(shape=(seq_length,))
    lstm_embedding = Embedding(vocab_size, 64, input_length=seq_length)(lstm_inputs)
    lstm_layer = LSTM(64, return_sequences=True)(lstm_embedding)
    lstm_pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm_layer)
    lstm_dense = Dense(64, activation='relu')(lstm_pooled)
    lstm_dropout = Dropout(0.5)(lstm_dense)
    lstm_outputs = Dense(1, activation='sigmoid')(lstm_dropout)
    
    lstm_model = Model(inputs=lstm_inputs, outputs=lstm_outputs)
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
    
    # Train both models
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print("Training Attention model...")
    attention_history = attention_model.fit(
        X_train, y_train, epochs=30, batch_size=32, 
        validation_split=0.2, callbacks=[early_stopping], verbose=0
    )
    
    print("Training LSTM model...")
    lstm_history = lstm_model.fit(
        X_train, y_train, epochs=30, batch_size=32, 
        validation_split=0.2, callbacks=[early_stopping], verbose=0
    )
    
    # Evaluate models
    attn_loss, attn_acc = attention_model.evaluate(X_test, y_test, verbose=0)
    lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nModel Comparison:")
    print(f"Attention Model: Accuracy = {attn_acc:.4f}, Loss = {attn_loss:.4f}")
    print(f"LSTM Model: Accuracy = {lstm_acc:.4f}, Loss = {lstm_loss:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(attention_history.history['val_accuracy'], label='Attention', color='blue')
    plt.plot(lstm_history.history['val_accuracy'], label='LSTM', color='red')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(attention_history.history['val_loss'], label='Attention', color='blue')
    plt.plot(lstm_history.history['val_loss'], label='LSTM', color='red')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run all attention examples"""
    print("=== ATTENTION MECHANISMS DEMONSTRATION ===\n")
    
    # 1. Scaled Dot-Product Attention
    demonstrate_scaled_dot_product_attention()
    
    # 2. Multi-Head Attention
    demonstrate_multi_head_attention()
    
    # 3. Self-Attention
    demonstrate_self_attention()
    
    # 4. Train model with attention
    train_attention_model()
    
    # 5. Attention analysis
    attention_analysis_example()
    
    # 6. Compare with LSTM
    compare_attention_with_lstm()
    
    print("\n" + "="*60)
    print("=== ATTENTION MECHANISMS COMPLETE! ===")
    print("="*60)

if __name__ == "__main__":
    main() 