"""
CNN Image Classification
========================

This script demonstrates Convolutional Neural Networks (CNN) for image classification including:
- Basic CNN architecture
- Data augmentation
- Transfer learning with pre-trained models
- CNN visualization techniques
- Different CNN architectures comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.utils import to_categorical
import cv2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_synthetic_image_data(n_samples=1000, img_size=32, n_classes=3):
    """Create synthetic image data for CNN demonstration"""
    print("Creating synthetic image data...")
    
    # Create synthetic images with different patterns
    X = np.zeros((n_samples, img_size, img_size, 3))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Create random patterns for different classes
        if i < n_samples // n_classes:
            # Class 0: Horizontal lines
            img = np.random.rand(img_size, img_size, 3) * 0.3
            for j in range(0, img_size, 4):
                img[j:j+2, :, :] = 1.0
            y[i] = 0
        elif i < 2 * n_samples // n_classes:
            # Class 1: Vertical lines
            img = np.random.rand(img_size, img_size, 3) * 0.3
            for j in range(0, img_size, 4):
                img[:, j:j+2, :] = 1.0
            y[i] = 1
        else:
            # Class 2: Diagonal lines
            img = np.random.rand(img_size, img_size, 3) * 0.3
            for j in range(0, img_size, 2):
                if j < img_size and j < img_size:
                    img[j, j, :] = 1.0
            y[i] = 2
        
        X[i] = img
    
    print(f"Synthetic image data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y

def create_mnist_like_data(n_samples=1000, img_size=28):
    """Create MNIST-like digit data"""
    print("Creating MNIST-like digit data...")
    
    # Generate random "digits" with different patterns
    X = np.zeros((n_samples, img_size, img_size, 1))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Create a blank image
        img = np.zeros((img_size, img_size))
        
        # Generate random digit pattern (0-9)
        digit = i % 10
        y[i] = digit
        
        # Create simple digit patterns
        if digit == 0:
            # Circle pattern
            center = img_size // 2
            radius = img_size // 4
            for x in range(img_size):
                for y_coord in range(img_size):
                    if (x - center)**2 + (y_coord - center)**2 <= radius**2:
                        img[x, y_coord] = 1.0
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
    
    print(f"MNIST-like data shape: X={X.shape}, y={y.shape}")
    print(f"Digit distribution: {np.bincount(y)}")
    return X, y

def build_basic_cnn(input_shape, num_classes):
    """Build a basic CNN architecture"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def build_deep_cnn(input_shape, num_classes):
    """Build a deeper CNN architecture"""
    model = Sequential([
        # First block
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fourth block
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def build_transfer_learning_model(base_model_name='VGG16', input_shape=(224, 224, 3), num_classes=3):
    """Build a transfer learning model using pre-trained networks"""
    
    if base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create new model on top
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def create_data_augmentation():
    """Create data augmentation pipeline"""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    return datagen

def visualize_cnn_filters(model, layer_name, num_filters=16):
    """Visualize CNN filters"""
    print(f"Visualizing filters from layer: {layer_name}")
    
    # Get the layer
    layer = model.get_layer(layer_name)
    
    # Get the filters
    filters, biases = layer.get_weights()
    
    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot filters
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(min(num_filters, filters.shape[3])):
        row, col = i // 4, i % 4
        filter_img = filters[:, :, 0, i]  # First channel
        axes[row, col].imshow(filter_img, cmap='gray')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle(f'CNN Filters from {layer_name}')
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, layer_name, sample_image, num_features=16):
    """Visualize feature maps from a specific layer"""
    print(f"Visualizing feature maps from layer: {layer_name}")
    
    # Create a model that outputs the feature maps
    feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    # Get feature maps
    feature_maps = feature_model.predict(sample_image)
    
    # Plot feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(min(num_features, feature_maps.shape[3])):
        row, col = i // 4, i % 4
        feature_map = feature_maps[0, :, :, i]
        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Feature {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle(f'Feature Maps from {layer_name}')
    plt.tight_layout()
    plt.show()

def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Training and Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Training and Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
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

def synthetic_image_classification():
    """Example: Synthetic image classification with CNN"""
    print("="*60)
    print("SYNTHETIC IMAGE CLASSIFICATION")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_image_data(1500, 32, 3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build and train basic CNN
    model = build_basic_cnn((32, 32, 3), 3)
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot results
    plot_training_history(history, "Basic CNN")
    plot_confusion_matrix(y_test, y_pred_classes, ['Horizontal', 'Vertical', 'Diagonal'])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=['Horizontal', 'Vertical', 'Diagonal']))
    
    return model, history

def mnist_like_classification():
    """Example: MNIST-like digit classification"""
    print("\n" + "="*60)
    print("MNIST-LIKE DIGIT CLASSIFICATION")
    print("="*60)
    
    # Create MNIST-like data
    X, y = create_mnist_like_data(2000, 28)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build deep CNN
    model = build_deep_cnn((28, 28, 1), 10)
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Plot results
    plot_training_history(history, "Deep CNN")
    plot_confusion_matrix(y_test, y_pred_classes, [str(i) for i in range(10)])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    return model, history

def data_augmentation_example():
    """Example: Data augmentation with CNN"""
    print("\n" + "="*60)
    print("DATA AUGMENTATION EXAMPLE")
    print("="*60)
    
    # Create synthetic data
    X, y = create_synthetic_image_data(1000, 32, 3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    
    # Create data augmentation
    datagen = create_data_augmentation()
    datagen.fit(X_train)
    
    # Build model
    model = build_basic_cnn((32, 32, 3), 3)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train with data augmentation
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test_cat),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy with Augmentation: {test_accuracy:.4f}")
    print(f"Test Loss with Augmentation: {test_loss:.4f}")
    
    # Plot results
    plot_training_history(history, "CNN with Data Augmentation")
    
    return model, history

def cnn_visualization_example():
    """Example: CNN visualization techniques"""
    print("\n" + "="*60)
    print("CNN VISUALIZATION EXAMPLE")
    print("="*60)
    
    # Create simple data
    X, y = create_synthetic_image_data(100, 32, 3)
    
    # Build model
    model = build_basic_cnn((32, 32, 3), 3)
    
    # Train model briefly
    y_cat = to_categorical(y, 3)
    model.fit(X, y_cat, epochs=10, batch_size=32, verbose=0)
    
    # Visualize filters
    visualize_cnn_filters(model, 'conv2d', 16)
    
    # Visualize feature maps
    sample_image = X[0:1]  # First image
    visualize_feature_maps(model, 'conv2d', sample_image, 16)
    
    print("CNN visualization complete!")

def transfer_learning_example():
    """Example: Transfer learning with pre-trained models"""
    print("\n" + "="*60)
    print("TRANSFER LEARNING EXAMPLE")
    print("="*60)
    
    # Create synthetic data (resize to 224x224 for pre-trained models)
    X, y = create_synthetic_image_data(500, 224, 3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    
    # Build transfer learning model
    model = build_transfer_learning_model('VGG16', (224, 224, 3), 3)
    
    print("Transfer learning model summary:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train_cat,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTransfer Learning Test Accuracy: {test_accuracy:.4f}")
    print(f"Transfer Learning Test Loss: {test_loss:.4f}")
    
    # Plot results
    plot_training_history(history, "Transfer Learning (VGG16)")
    
    return model, history

def cnn_architecture_comparison():
    """Compare different CNN architectures"""
    print("\n" + "="*60)
    print("CNN ARCHITECTURE COMPARISON")
    print("="*60)
    
    # Create data
    X, y = create_synthetic_image_data(1000, 32, 3)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Convert to categorical
    y_train_cat = to_categorical(y_train, 3)
    y_test_cat = to_categorical(y_test, 3)
    
    # Build different models
    models = {
        'Basic CNN': build_basic_cnn((32, 32, 3), 3),
        'Deep CNN': build_deep_cnn((32, 32, 3), 3)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train_cat,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        
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

def main():
    """Main function to run all CNN examples"""
    print("=== CNN IMAGE CLASSIFICATION EXAMPLES ===\n")
    
    # 1. Synthetic image classification
    synthetic_image_classification()
    
    # 2. MNIST-like digit classification
    mnist_like_classification()
    
    # 3. Data augmentation example
    data_augmentation_example()
    
    # 4. CNN visualization
    cnn_visualization_example()
    
    # 5. Transfer learning
    transfer_learning_example()
    
    # 6. Architecture comparison
    cnn_architecture_comparison()
    
    print("\n" + "="*60)
    print("=== CNN EXAMPLES COMPLETE! ===")
    print("="*60)

if __name__ == "__main__":
    main() 