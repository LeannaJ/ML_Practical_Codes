"""
Image Classification Examples
============================

- CNN from Scratch
- Transfer Learning (ResNet, VGG, EfficientNet)
- Data Augmentation
- Model Evaluation and Visualization
- Multi-class Classification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# For deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, 
                                       Dropout, BatchNormalization, GlobalAveragePooling2D)
    from tensorflow.keras.applications import (ResNet50, VGG16, EfficientNetB0, 
                                             MobileNetV2, InceptionV3)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# For image processing
try:
    from PIL import Image
    import cv2
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    print("OpenCV/PIL not available. Install with: pip install opencv-python pillow")

# 1. Generate Synthetic Image Data
print("=== Image Classification Examples ===")

def generate_synthetic_images(n_samples=2000, img_size=64, n_classes=5):
    """Generate synthetic image data for classification"""
    np.random.seed(42)
    
    # Define classes
    classes = ['circle', 'square', 'triangle', 'star', 'cross']
    
    images = []
    labels = []
    
    for i in range(n_samples):
        # Create blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Randomly select class
        class_idx = np.random.randint(0, n_classes)
        class_name = classes[class_idx]
        
        # Generate shape based on class
        if class_name == 'circle':
            # Draw circle
            center = (np.random.randint(20, img_size-20), np.random.randint(20, img_size-20))
            radius = np.random.randint(10, 20)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            cv2.circle(img, center, radius, color, -1)
            
        elif class_name == 'square':
            # Draw square
            x = np.random.randint(10, img_size-30)
            y = np.random.randint(10, img_size-30)
            size = np.random.randint(15, 25)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            cv2.rectangle(img, (x, y), (x+size, y+size), color, -1)
            
        elif class_name == 'triangle':
            # Draw triangle
            pts = np.array([[np.random.randint(10, img_size-10), np.random.randint(10, img_size//2)],
                           [np.random.randint(10, img_size//2), np.random.randint(img_size//2, img_size-10)],
                           [np.random.randint(img_size//2, img_size-10), np.random.randint(img_size//2, img_size-10)]], np.int32)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            cv2.fillPoly(img, [pts], color)
            
        elif class_name == 'star':
            # Draw star (simplified as multiple triangles)
            center = (np.random.randint(20, img_size-20), np.random.randint(20, img_size-20))
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            for angle in range(0, 360, 72):
                x1 = center[0] + int(15 * np.cos(np.radians(angle)))
                y1 = center[1] + int(15 * np.sin(np.radians(angle)))
                x2 = center[0] + int(8 * np.cos(np.radians(angle + 36)))
                y2 = center[1] + int(8 * np.sin(np.radians(angle + 36)))
                x3 = center[0] + int(15 * np.cos(np.radians(angle + 72)))
                y3 = center[1] + int(15 * np.sin(np.radians(angle + 72)))
                pts = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
                cv2.fillPoly(img, [pts], color)
                
        else:  # cross
            # Draw cross
            center = (np.random.randint(20, img_size-20), np.random.randint(20, img_size-20))
            size = np.random.randint(8, 15)
            color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
            cv2.line(img, (center[0]-size, center[1]), (center[0]+size, center[1]), color, 3)
            cv2.line(img, (center[0], center[1]-size), (center[0], center[1]+size), color, 3)
        
        # Add some noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255)
        
        images.append(img)
        labels.append(class_name)
    
    return np.array(images), np.array(labels)

# Generate synthetic image data
print("Generating synthetic image data...")
X_images, y_labels = generate_synthetic_images()
print(f"Image data shape: {X_images.shape}")
print(f"Number of classes: {len(np.unique(y_labels))}")
print(f"Classes: {np.unique(y_labels)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(np.unique(y_labels)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Normalize images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 2. CNN from Scratch
print("\n=== CNN from Scratch ===")

def create_cnn_model(input_shape, num_classes):
    """Create a CNN model from scratch"""
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

if TENSORFLOW_AVAILABLE:
    # Create and compile CNN model
    cnn_model = create_cnn_model(X_train.shape[1:], len(np.unique(y_labels)))
    cnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("CNN Model Summary:")
    cnn_model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train CNN model
    print("\nTraining CNN model...")
    cnn_history = cnn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate CNN model
    cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nCNN Test Accuracy: {cnn_accuracy:.4f}")
    print(f"CNN Test Loss: {cnn_loss:.4f}")

# 3. Transfer Learning
print("\n=== Transfer Learning ===")

def create_transfer_learning_model(base_model_name, input_shape, num_classes):
    """Create a transfer learning model using pre-trained networks"""
    
    if base_model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

if TENSORFLOW_AVAILABLE:
    # Create transfer learning models
    transfer_models = {}
    model_names = ['ResNet50', 'VGG16', 'EfficientNetB0', 'MobileNetV2']
    
    for model_name in model_names:
        print(f"\nCreating {model_name} transfer learning model...")
        try:
            model = create_transfer_learning_model(model_name, X_train.shape[1:], len(np.unique(y_labels)))
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            transfer_models[model_name] = model
        except Exception as e:
            print(f"Error creating {model_name}: {e}")
            continue
    
    # Train transfer learning models
    transfer_histories = {}
    for model_name, model in transfer_models.items():
        print(f"\nTraining {model_name}...")
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        transfer_histories[model_name] = history
        
        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")

# 4. Data Augmentation
print("\n=== Data Augmentation ===")

if TENSORFLOW_AVAILABLE:
    # Create data augmentation generator
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
    
    # Create model with data augmentation
    aug_model = create_cnn_model(X_train.shape[1:], len(np.unique(y_labels)))
    aug_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model with data augmentation...")
    aug_history = aug_model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=30,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate augmented model
    aug_loss, aug_accuracy = aug_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAugmented Model Test Accuracy: {aug_accuracy:.4f}")

# 5. Model Evaluation and Visualization
print("\n=== Model Evaluation and Visualization ===")

if TENSORFLOW_AVAILABLE:
    # Get predictions
    cnn_predictions = cnn_model.predict(X_test)
    cnn_pred_classes = np.argmax(cnn_predictions, axis=1)
    cnn_true_classes = np.argmax(y_test, axis=1)
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot training history
    axes[0, 0].plot(cnn_history.history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('CNN Training History')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss history
    axes[0, 1].plot(cnn_history.history['loss'], label='Training Loss')
    axes[0, 1].plot(cnn_history.history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('CNN Loss History')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(cnn_true_classes, cnn_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title('Confusion Matrix')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('True')
    
    # Sample predictions
    sample_indices = np.random.choice(len(X_test), 9, replace=False)
    for i, idx in enumerate(sample_indices):
        row = i // 3
        col = i % 3
        axes[1, row].imshow(X_test[idx])
        pred_class = le.inverse_transform([cnn_pred_classes[idx]])[0]
        true_class = le.inverse_transform([cnn_true_classes[idx]])[0]
        color = 'green' if pred_class == true_class else 'red'
        axes[1, row].set_title(f'Pred: {pred_class}\nTrue: {true_class}', color=color)
        axes[1, row].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Model comparison
    if transfer_models:
        model_names = ['CNN'] + list(transfer_models.keys())
        accuracies = [cnn_accuracy]
        
        for model_name, model in transfer_models.items():
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            accuracies.append(accuracy)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, accuracies, alpha=0.8)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# 6. Feature Visualization
print("\n=== Feature Visualization ===")

if TENSORFLOW_AVAILABLE:
    # Get intermediate layer outputs
    layer_outputs = [layer.output for layer in cnn_model.layers if 'conv' in layer.name]
    activation_model = Model(inputs=cnn_model.input, outputs=layer_outputs)
    
    # Get activations for a sample image
    sample_image = X_test[0:1]
    activations = activation_model.predict(sample_image)
    
    # Visualize first few feature maps
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(min(4, len(activations))):
        # First row: original image
        if i == 0:
            axes[0, i].imshow(sample_image[0])
            axes[0, i].set_title('Original Image')
        else:
            axes[0, i].axis('off')
        
        # Second row: feature maps
        feature_maps = activations[i][0]
        # Show first 4 feature maps
        for j in range(min(4, feature_maps.shape[-1])):
            axes[1, i].imshow(feature_maps[:, :, j], cmap='viridis')
            axes[1, i].set_title(f'Feature Map {j+1}')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 7. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

if TENSORFLOW_AVAILABLE:
    print("Image Classification Results:")
    print(f"1. CNN from Scratch:")
    print(f"   - Test Accuracy: {cnn_accuracy:.4f}")
    print(f"   - Test Loss: {cnn_loss:.4f}")
    
    if transfer_models:
        print(f"\n2. Transfer Learning Models:")
        for model_name, model in transfer_models.items():
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"   - {model_name}: {accuracy:.4f}")
    
    print(f"\n3. Data Augmentation:")
    print(f"   - Augmented Model Accuracy: {aug_accuracy:.4f}")

print(f"\nKey Insights:")
print(f"- CNN models can effectively learn spatial patterns in images")
print(f"- Transfer learning provides good performance with less training data")
print(f"- Data augmentation helps improve model generalization")
print(f"- Batch normalization and dropout help prevent overfitting")

print(f"\nRecommendations:")
print(f"- Use CNN for image classification tasks")
print(f"- Apply transfer learning for small datasets")
print(f"- Use data augmentation to improve model robustness")
print(f"- Monitor training/validation curves to prevent overfitting")
print(f"- Use appropriate evaluation metrics (accuracy, confusion matrix)")
print(f"- Consider model complexity vs performance trade-offs")
print(f"- Experiment with different architectures and hyperparameters") 