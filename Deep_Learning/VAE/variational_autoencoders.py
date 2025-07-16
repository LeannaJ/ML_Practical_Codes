"""
Variational Autoencoders (VAE)
==============================

This script demonstrates Variational Autoencoders including:
- Basic VAE implementation
- VAE for image generation
- VAE for dimensionality reduction
- Conditional VAE
- VAE with different architectures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_synthetic_image_data(n_samples=1000, img_size=28):
    """Create synthetic image data for VAE demonstration"""
    print("Creating synthetic image data...")
    
    # Create synthetic images with different patterns
    X = np.zeros((n_samples, img_size, img_size, 1))
    
    for i in range(n_samples):
        # Create different patterns
        if i < n_samples // 4:
            # Horizontal lines
            img = np.random.rand(img_size, img_size) * 0.1
            for j in range(0, img_size, 4):
                img[j:j+2, :] = 1.0
        elif i < n_samples // 2:
            # Vertical lines
            img = np.random.rand(img_size, img_size) * 0.1
            for j in range(0, img_size, 4):
                img[:, j:j+2] = 1.0
        elif i < 3 * n_samples // 4:
            # Circles
            img = np.random.rand(img_size, img_size) * 0.1
            center = img_size // 2
            radius = img_size // 4
            for x in range(img_size):
                for y in range(img_size):
                    if (x - center)**2 + (y - center)**2 <= radius**2:
                        img[x, y] = 1.0
        else:
            # Random patterns
            img = np.random.rand(img_size, img_size) > 0.5
        
        X[i, :, :, 0] = img
    
    print(f"Synthetic image data shape: {X.shape}")
    return X

def create_mnist_like_data(n_samples=1000, img_size=28):
    """Create MNIST-like digit data"""
    print("Creating MNIST-like digit data...")
    
    # Generate random "digits" with different patterns
    X = np.zeros((n_samples, img_size, img_size, 1))
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Create a blank image
        img = np.zeros((img_size, img_size))
        
        # Generate random digit pattern (0-9)
        digit = i % 10
        labels[i] = digit
        
        # Create simple digit patterns
        if digit == 0:
            # Circle pattern
            center = img_size // 2
            radius = img_size // 4
            for x in range(img_size):
                for y in range(img_size):
                    if (x - center)**2 + (y - center)**2 <= radius**2:
                        img[x, y] = 1.0
        elif digit == 1:
            # Vertical line
            img[:, img_size//2-1:img_size//2+1] = 1.0
        elif digit == 2:
            # Z pattern
            img[0, :] = 1.0
            img[-1, :] = 1.0
            for j in range(img_size):
                img[j, img_size-1-j] = 1.0
        else:
            # Random pattern for other digits
            img = np.random.rand(img_size, img_size) > 0.7
        
        X[i, :, :, 0] = img
    
    print(f"MNIST-like data shape: {X.shape}")
    print(f"Digit distribution: {np.bincount(labels)}")
    return X, labels

def create_high_dimensional_data(n_samples=1000, n_features=100):
    """Create high-dimensional data for VAE dimensionality reduction"""
    print("Creating high-dimensional data...")
    
    # Generate high-dimensional data with underlying structure
    np.random.seed(42)
    
    # Create latent factors
    n_factors = 10
    factors = np.random.randn(n_samples, n_factors)
    
    # Create mixing matrix
    mixing_matrix = np.random.randn(n_factors, n_features)
    
    # Generate high-dimensional data
    X = np.dot(factors, mixing_matrix) + 0.1 * np.random.randn(n_samples, n_features)
    
    # Add some noise
    X += 0.05 * np.random.randn(n_samples, n_features)
    
    print(f"High-dimensional data shape: {X.shape}")
    return X, factors

def sampling(args):
    """Sampling function for VAE"""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_basic_vae(input_shape, latent_dim=2):
    """Build a basic VAE"""
    # Encoder
    inputs = Input(shape=input_shape)
    
    if len(input_shape) == 1:  # 1D input (tabular data)
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x_flat = x
    else:  # 2D input (image data)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x_flat = Flatten()(x)
        x_flat = Dense(128, activation='relu')(x_flat)
    
    # Latent space
    z_mean = Dense(latent_dim, name='z_mean')(x_flat)
    z_log_var = Dense(latent_dim, name='z_log_var')(x_flat)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Decoder
    if len(input_shape) == 1:  # 1D input
        x = Dense(32, activation='relu')(z)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(input_shape[0], activation='sigmoid')(x)
    else:  # 2D input
        x = Dense(128, activation='relu')(z)
        x = Dense(7 * 7 * 64, activation='relu')(x)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create models
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(z, outputs, name='decoder')
    vae = Model(inputs, outputs, name='vae')
    
    # VAE loss
    def vae_loss(inputs, outputs):
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return reconstruction_loss + 0.1 * kl_loss
    
    vae.compile(optimizer=Adam(learning_rate=0.001), loss=vae_loss)
    
    return vae, encoder, decoder

def build_conv_vae(input_shape, latent_dim=2):
    """Build a convolutional VAE for images"""
    # Encoder
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Flatten
    x_flat = Flatten()(x)
    x_flat = Dense(256, activation='relu')(x_flat)
    
    # Latent space
    z_mean = Dense(latent_dim, name='z_mean')(x_flat)
    z_log_var = Dense(latent_dim, name='z_log_var')(x_flat)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Decoder
    x = Dense(256, activation='relu')(z)
    x = Dense(7 * 7 * 128, activation='relu')(x)
    x = Reshape((7, 7, 128))(x)
    
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create models
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = Model(z, outputs, name='decoder')
    vae = Model(inputs, outputs, name='vae')
    
    # VAE loss
    def vae_loss(inputs, outputs):
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return reconstruction_loss + 0.1 * kl_loss
    
    vae.compile(optimizer=Adam(learning_rate=0.001), loss=vae_loss)
    
    return vae, encoder, decoder

def build_conditional_vae(input_shape, num_classes, latent_dim=2):
    """Build a conditional VAE"""
    # Input layers
    inputs = Input(shape=input_shape)
    condition = Input(shape=(num_classes,))
    
    # Combine input and condition
    if len(input_shape) == 1:
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x_flat = x
    else:
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x_flat = Flatten()(x)
        x_flat = Dense(128, activation='relu')(x_flat)
    
    # Concatenate with condition
    x_cond = tf.keras.layers.concatenate([x_flat, condition])
    
    # Latent space
    z_mean = Dense(latent_dim, name='z_mean')(x_cond)
    z_log_var = Dense(latent_dim, name='z_log_var')(x_cond)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Concatenate z with condition for decoder
    z_cond = tf.keras.layers.concatenate([z, condition])
    
    # Decoder
    if len(input_shape) == 1:
        x = Dense(32, activation='relu')(z_cond)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(input_shape[0], activation='sigmoid')(x)
    else:
        x = Dense(128, activation='relu')(z_cond)
        x = Dense(7 * 7 * 64, activation='relu')(x)
        x = Reshape((7, 7, 64))(x)
        x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        outputs = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    # Create models
    encoder = Model([inputs, condition], [z_mean, z_log_var, z], name='encoder')
    decoder = Model([z, condition], outputs, name='decoder')
    vae = Model([inputs, condition], outputs, name='vae')
    
    # VAE loss
    def vae_loss(inputs, outputs):
        reconstruction_loss = K.mean(K.square(inputs - outputs))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return reconstruction_loss + 0.1 * kl_loss
    
    vae.compile(optimizer=Adam(learning_rate=0.001), loss=vae_loss)
    
    return vae, encoder, decoder

def plot_training_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(10, 4))
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_latent_space(encoder, X, labels=None, model_name="VAE"):
    """Plot latent space representation"""
    print(f"Visualizing latent space for {model_name}...")
    
    # Get latent representations
    z_mean, z_log_var, z = encoder.predict(X)
    
    # Plot latent space
    plt.figure(figsize=(15, 5))
    
    # 2D scatter plot
    plt.subplot(1, 3, 1)
    if labels is not None:
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(z[:, 0], z[:, 1], alpha=0.6)
    plt.title(f'{model_name} - Latent Space')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True)
    
    # Distribution of latent variables
    plt.subplot(1, 3, 2)
    plt.hist(z[:, 0], bins=30, alpha=0.7, label='z[0]')
    plt.hist(z[:, 1], bins=30, alpha=0.7, label='z[1]')
    plt.title('Latent Variable Distributions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # Correlation between latent variables
    plt.subplot(1, 3, 3)
    plt.scatter(z[:, 0], z[:, 1], alpha=0.6)
    plt.title('Latent Variable Correlation')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return z

def plot_reconstructions(original, reconstructed, n_samples=8, model_name="VAE"):
    """Plot original vs reconstructed samples"""
    print(f"Visualizing reconstructions for {model_name}...")
    
    # Select random samples
    indices = np.random.choice(len(original), n_samples, replace=False)
    
    # Reshape if needed
    if len(original.shape) == 4:  # Image data
        original_samples = original[indices]
        reconstructed_samples = reconstructed[indices]
        
        # Plot
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        for i in range(n_samples):
            # Original
            axes[0, i].imshow(original_samples[i, :, :, 0], cmap='gray')
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Reconstructed
            axes[1, i].imshow(reconstructed_samples[i, :, :, 0], cmap='gray')
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].axis('off')
        
        plt.suptitle(f'{model_name} - Original vs Reconstructed')
        plt.tight_layout()
        plt.show()
    else:  # Tabular data
        original_samples = original[indices]
        reconstructed_samples = reconstructed[indices]
        
        # Plot
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        
        for i in range(n_samples):
            # Original
            axes[0, i].plot(original_samples[i])
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].grid(True)
            
            # Reconstructed
            axes[1, i].plot(reconstructed_samples[i])
            axes[1, i].set_title(f'Reconstructed {i+1}')
            axes[1, i].grid(True)
        
        plt.suptitle(f'{model_name} - Original vs Reconstructed')
        plt.tight_layout()
        plt.show()

def plot_generated_samples(decoder, latent_dim=2, n_samples=16, model_name="VAE"):
    """Plot generated samples from decoder"""
    print(f"Generating samples from {model_name}...")
    
    # Generate random latent vectors
    z = np.random.normal(0, 1, (n_samples, latent_dim))
    
    # Generate samples
    generated = decoder.predict(z)
    
    # Plot generated samples
    if len(generated.shape) == 4:  # Image data
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for i in range(n_samples):
            row, col = i // 4, i % 4
            axes[row, col].imshow(generated[i, :, :, 0], cmap='gray')
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].axis('off')
        
        plt.suptitle(f'{model_name} - Generated Samples')
        plt.tight_layout()
        plt.show()
    else:  # Tabular data
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        
        for i in range(n_samples):
            row, col = i // 4, i % 4
            axes[row, col].plot(generated[i])
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].grid(True)
        
        plt.suptitle(f'{model_name} - Generated Samples')
        plt.tight_layout()
        plt.show()

def image_vae_example():
    """Example: VAE for image generation"""
    print("="*60)
    print("VAE IMAGE GENERATION EXAMPLE")
    print("="*60)
    
    # Create synthetic image data
    X = create_synthetic_image_data(1000, 28)
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build VAE
    vae, encoder, decoder = build_conv_vae((28, 28, 1), latent_dim=2)
    
    print("VAE model summary:")
    vae.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train VAE
    history = vae.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate VAE
    test_loss = vae.evaluate(X_test, X_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Generate reconstructions
    reconstructed = vae.predict(X_test)
    
    # Plot results
    plot_training_history(history, "Image VAE")
    plot_latent_space(encoder, X_test, model_name="Image VAE")
    plot_reconstructions(X_test, reconstructed, model_name="Image VAE")
    plot_generated_samples(decoder, latent_dim=2, model_name="Image VAE")
    
    return vae, encoder, decoder, history

def tabular_vae_example():
    """Example: VAE for tabular data"""
    print("\n" + "="*60)
    print("VAE TABULAR DATA EXAMPLE")
    print("="*60)
    
    # Create high-dimensional data
    X, true_factors = create_high_dimensional_data(1000, 100)
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Build VAE
    vae, encoder, decoder = build_basic_vae((X_train_scaled.shape[1],), latent_dim=10)
    
    print("VAE model summary:")
    vae.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train VAE
    history = vae.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate VAE
    test_loss = vae.evaluate(X_test_scaled, X_test_scaled, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Generate reconstructions
    reconstructed = vae.predict(X_test_scaled)
    
    # Calculate reconstruction error
    mse = mean_squared_error(X_test_scaled, reconstructed)
    print(f"Reconstruction MSE: {mse:.4f}")
    
    # Plot results
    plot_training_history(history, "Tabular VAE")
    plot_latent_space(encoder, X_test_scaled, model_name="Tabular VAE")
    plot_reconstructions(X_test_scaled, reconstructed, model_name="Tabular VAE")
    plot_generated_samples(decoder, latent_dim=10, model_name="Tabular VAE")
    
    return vae, encoder, decoder, history

def conditional_vae_example():
    """Example: Conditional VAE"""
    print("\n" + "="*60)
    print("CONDITIONAL VAE EXAMPLE")
    print("="*60)
    
    # Create MNIST-like data with labels
    X, labels = create_mnist_like_data(1000, 28)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)
    
    # Build conditional VAE
    vae, encoder, decoder = build_conditional_vae((28, 28, 1), 10, latent_dim=2)
    
    print("Conditional VAE model summary:")
    vae.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train VAE
    history = vae.fit(
        [X_train, y_train_onehot], X_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate VAE
    test_loss = vae.evaluate([X_test, y_test_onehot], X_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    
    # Generate reconstructions
    reconstructed = vae.predict([X_test, y_test_onehot])
    
    # Plot results
    plot_training_history(history, "Conditional VAE")
    plot_latent_space(encoder, [X_test, y_test_onehot], y_test, model_name="Conditional VAE")
    plot_reconstructions(X_test, reconstructed, model_name="Conditional VAE")
    
    # Generate samples for specific conditions
    print("\nGenerating samples for specific digits...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for digit in range(5):
        # Generate condition for this digit
        condition = np.zeros((1, 10))
        condition[0, digit] = 1
        
        # Generate samples
        z = np.random.normal(0, 1, (1, 2))
        generated = decoder.predict([z, condition])
        
        # Plot
        axes[0, digit].imshow(generated[0, :, :, 0], cmap='gray')
        axes[0, digit].set_title(f'Digit {digit}')
        axes[0, digit].axis('off')
        
        # Generate another sample
        z = np.random.normal(0, 1, (1, 2))
        generated = decoder.predict([z, condition])
        
        axes[1, digit].imshow(generated[0, :, :, 0], cmap='gray')
        axes[1, digit].set_title(f'Digit {digit} (2nd)')
        axes[1, digit].axis('off')
    
    plt.suptitle('Conditional VAE - Generated Samples for Specific Digits')
    plt.tight_layout()
    plt.show()
    
    return vae, encoder, decoder, history

def vae_architecture_comparison():
    """Compare different VAE architectures"""
    print("\n" + "="*60)
    print("VAE ARCHITECTURE COMPARISON")
    print("="*60)
    
    # Create synthetic image data
    X = create_synthetic_image_data(800, 28)
    
    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Build different VAE architectures
    models = {
        'Basic VAE': build_basic_vae((28, 28, 1), latent_dim=2),
        'Conv VAE': build_conv_vae((28, 28, 1), latent_dim=2)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, (vae, encoder, decoder) in models.items():
        print(f"\nTraining {name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train model
        history = vae.fit(
            X_train, X_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        test_loss = vae.evaluate(X_test, X_test, verbose=0)
        
        results[name] = {
            'loss': test_loss,
            'history': history,
            'encoder': encoder,
            'decoder': decoder
        }
        
        print(f"{name} - Test Loss: {test_loss:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history'].history['loss'], label=f'{name} (Train)')
        plt.plot(result['history'].history['val_loss'], label=f'{name} (Val)')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    losses = [results[name]['loss'] for name in names]
    plt.bar(names, losses)
    plt.title('Final Test Loss Comparison')
    plt.ylabel('Test Loss')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot latent spaces
    plt.figure(figsize=(12, 4))
    
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(1, 2, i+1)
        z_mean, z_log_var, z = result['encoder'].predict(X_test)
        plt.scatter(z[:, 0], z[:, 1], alpha=0.6)
        plt.title(f'{name} - Latent Space')
        plt.xlabel('z[0]')
        plt.ylabel('z[1]')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    for name, result in results.items():
        print(f"{name}: Test Loss = {result['loss']:.4f}")

def main():
    """Main function to run all VAE examples"""
    print("=== VARIATIONAL AUTOENCODERS EXAMPLES ===\n")
    
    # 1. Image VAE
    image_vae_example()
    
    # 2. Tabular VAE
    tabular_vae_example()
    
    # 3. Conditional VAE
    conditional_vae_example()
    
    # 4. Architecture comparison
    vae_architecture_comparison()
    
    print("\n" + "="*60)
    print("=== VAE EXAMPLES COMPLETE! ===")
    print("="*60)

if __name__ == "__main__":
    main() 