"""
Image Segmentation Examples
===========================

- U-Net Architecture
- Fully Convolutional Networks (FCN)
- Semantic Segmentation
- Instance Segmentation
- Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# For deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, 
                                       Flatten, Dropout, BatchNormalization,
                                       UpSampling2D, Concatenate, Conv2DTranspose)
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

# 1. Generate Synthetic Segmentation Data
print("=== Image Segmentation Examples ===")

def generate_synthetic_segmentation_data(n_samples=1000, img_size=128, n_classes=3):
    """Generate synthetic images with segmentation masks"""
    np.random.seed(42)
    
    # Define segmentation classes
    classes = ['background', 'object1', 'object2']
    class_colors = {
        'background': (0, 0, 0),      # Black
        'object1': (255, 0, 0),       # Red
        'object2': (0, 255, 0)        # Green
    }
    
    images = []
    masks = []
    
    for i in range(n_samples):
        # Create blank image and mask
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Add background noise
        img += np.random.randint(0, 50, img.shape, dtype=np.uint8)
        
        # Generate random number of objects
        n_objects = np.random.randint(1, 4)
        
        for j in range(n_objects):
            # Randomly select object class (skip background)
            class_name = np.random.choice(['object1', 'object2'])
            class_id = classes.index(class_name)
            color = class_colors[class_name]
            
            # Random position and size
            x = np.random.randint(20, img_size - 40)
            y = np.random.randint(20, img_size - 40)
            size = np.random.randint(15, 35)
            
            # Random shape type
            shape_type = np.random.choice(['circle', 'square', 'triangle'])
            
            # Draw object
            if shape_type == 'circle':
                cv2.circle(img, (x, y), size, color, -1)
                cv2.circle(mask, (x, y), size, class_id, -1)
                
            elif shape_type == 'square':
                cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), class_id, -1)
                
            else:  # triangle
                pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
                cv2.fillPoly(img, [pts], color)
                cv2.fillPoly(mask, [pts], class_id)
        
        # Add some texture and noise
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Generate synthetic segmentation data
print("Generating synthetic segmentation data...")
X_images, y_masks = generate_synthetic_segmentation_data()
print(f"Image data shape: {X_images.shape}")
print(f"Mask data shape: {y_masks.shape}")
print(f"Number of classes: {len(np.unique(y_masks))}")

# Convert masks to one-hot encoding
y_masks_onehot = tf.keras.utils.to_categorical(y_masks, num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y_masks_onehot, test_size=0.2, random_state=42
)

# Normalize images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 2. U-Net Architecture
print("\n=== U-Net Architecture ===")

def create_unet_model(input_shape, num_classes):
    """Create U-Net model for image segmentation"""
    
    def conv_block(inputs, filters, kernel_size=3):
        """Convolutional block with batch normalization and ReLU"""
        x = Conv2D(filters, kernel_size, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Encoder (downsampling path)
    # Level 1
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Level 2
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Level 3
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Level 4
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D((2, 2))(conv4)
    
    # Bridge
    conv5 = conv_block(pool4, 1024)
    
    # Decoder (upsampling path)
    # Level 4
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = Concatenate()([up6, conv4])
    conv6 = conv_block(up6, 512)
    
    # Level 3
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = Concatenate()([up7, conv3])
    conv7 = conv_block(up7, 256)
    
    # Level 2
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = Concatenate()([up8, conv2])
    conv8 = conv_block(up8, 128)
    
    # Level 1
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = Concatenate()([up9, conv1])
    conv9 = conv_block(up9, 64)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if TENSORFLOW_AVAILABLE:
    # Create and compile U-Net model
    unet_model = create_unet_model(X_train.shape[1:], num_classes=3)
    unet_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("U-Net Model Summary:")
    unet_model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train U-Net model
    print("\nTraining U-Net model...")
    unet_history = unet_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate U-Net model
    unet_loss, unet_accuracy = unet_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nU-Net Test Accuracy: {unet_accuracy:.4f}")
    print(f"U-Net Test Loss: {unet_loss:.4f}")

# 3. Fully Convolutional Network (FCN)
print("\n=== Fully Convolutional Network (FCN) ===")

def create_fcn_model(input_shape, num_classes):
    """Create FCN model for semantic segmentation"""
    
    inputs = Input(shape=input_shape)
    
    # Encoder (VGG-like)
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Decoder (upsampling)
    # Upsample to original size
    x = Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if TENSORFLOW_AVAILABLE:
    # Create and compile FCN model
    fcn_model = create_fcn_model(X_train.shape[1:], num_classes=3)
    fcn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("FCN Model Summary:")
    fcn_model.summary()
    
    # Train FCN model
    print("\nTraining FCN model...")
    fcn_history = fcn_model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate FCN model
    fcn_loss, fcn_accuracy = fcn_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFCN Test Accuracy: {fcn_accuracy:.4f}")
    print(f"FCN Test Loss: {fcn_loss:.4f}")

# 4. Semantic Segmentation with Skip Connections
print("\n=== Semantic Segmentation with Skip Connections ===")

def create_skip_connection_model(input_shape, num_classes):
    """Create segmentation model with skip connections"""
    
    inputs = Input(shape=input_shape)
    
    # Encoder with skip connections
    # Level 1
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    
    # Level 2
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    # Level 3
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    
    # Bridge
    bridge = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    bridge = Conv2D(512, (3, 3), activation='relu', padding='same')(bridge)
    
    # Decoder with skip connections
    # Level 3
    up3 = UpSampling2D((2, 2))(bridge)
    up3 = Concatenate()([up3, conv3])
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    up3 = Conv2D(256, (3, 3), activation='relu', padding='same')(up3)
    
    # Level 2
    up2 = UpSampling2D((2, 2))(up3)
    up2 = Concatenate()([up2, conv2])
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    up2 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    
    # Level 1
    up1 = UpSampling2D((2, 2))(up2)
    up1 = Concatenate()([up1, conv1])
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    up1 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    
    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(up1)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if TENSORFLOW_AVAILABLE:
    # Create and compile skip connection model
    skip_model = create_skip_connection_model(X_train.shape[1:], num_classes=3)
    skip_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Skip Connection Model Summary:")
    skip_model.summary()
    
    # Train skip connection model
    print("\nTraining Skip Connection model...")
    skip_history = skip_model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate skip connection model
    skip_loss, skip_accuracy = skip_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nSkip Connection Test Accuracy: {skip_accuracy:.4f}")
    print(f"Skip Connection Test Loss: {skip_loss:.4f}")

# 5. Model Evaluation and Visualization
print("\n=== Model Evaluation and Visualization ===")

def calculate_iou(y_true, y_pred, num_classes):
    """Calculate Intersection over Union for each class"""
    ious = []
    
    for class_id in range(num_classes):
        # Extract binary masks for current class
        true_mask = y_true[:, :, :, class_id]
        pred_mask = y_pred[:, :, :, class_id]
        
        # Calculate IoU
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    return np.mean(ious)

def visualize_segmentation_results(model, X_test, y_test, num_samples=6):
    """Visualize segmentation results"""
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        true_mask = np.argmax(y_test[i], axis=-1)
        axes[i, 1].imshow(true_mask, cmap='viridis')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        pred = model.predict(X_test[i:i+1], verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)
        axes[i, 2].imshow(pred_mask, cmap='viridis')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = X_test[i].copy()
        overlay[pred_mask == 1] = [255, 0, 0]  # Red for object1
        overlay[pred_mask == 2] = [0, 255, 0]  # Green for object2
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

if TENSORFLOW_AVAILABLE:
    # Visualize results
    print("Visualizing U-Net results...")
    visualize_segmentation_results(unet_model, X_test, y_test)
    
    # Calculate IoU scores
    print("\nCalculating IoU scores...")
    
    # U-Net IoU
    unet_predictions = unet_model.predict(X_test, verbose=0)
    unet_iou = calculate_iou(y_test, unet_predictions, num_classes=3)
    print(f"U-Net Mean IoU: {unet_iou:.4f}")
    
    # FCN IoU
    fcn_predictions = fcn_model.predict(X_test, verbose=0)
    fcn_iou = calculate_iou(y_test, fcn_predictions, num_classes=3)
    print(f"FCN Mean IoU: {fcn_iou:.4f}")
    
    # Skip Connection IoU
    skip_predictions = skip_model.predict(X_test, verbose=0)
    skip_iou = calculate_iou(y_test, skip_predictions, num_classes=3)
    print(f"Skip Connection Mean IoU: {skip_iou:.4f}")
    
    # Model comparison
    models = ['U-Net', 'FCN', 'Skip Connection']
    accuracies = [unet_accuracy, fcn_accuracy, skip_accuracy]
    ious = [unet_iou, fcn_iou, skip_iou]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy comparison
    bars1 = axes[0].bar(models, accuracies, alpha=0.8, color=['blue', 'green', 'red'])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # IoU comparison
    bars2 = axes[1].bar(models, ious, alpha=0.8, color=['blue', 'green', 'red'])
    axes[1].set_title('Model IoU Comparison')
    axes[1].set_ylabel('Mean IoU')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, iou in zip(bars2, ious):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{iou:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# 6. Advanced Segmentation Techniques
print("\n=== Advanced Segmentation Techniques ===")

def create_deep_lab_model(input_shape, num_classes):
    """Create DeepLab-style model with atrous convolutions"""
    
    inputs = Input(shape=input_shape)
    
    # Encoder with atrous convolutions
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Atrous convolutions with different rates
    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=1)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', dilation_rate=4)(x)
    
    # Decoder
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

if TENSORFLOW_AVAILABLE:
    # Create and compile DeepLab-style model
    deeplab_model = create_deep_lab_model(X_train.shape[1:], num_classes=3)
    deeplab_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("DeepLab-style Model Summary:")
    deeplab_model.summary()
    
    # Train DeepLab-style model
    print("\nTraining DeepLab-style model...")
    deeplab_history = deeplab_model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate DeepLab-style model
    deeplab_loss, deeplab_accuracy = deeplab_model.evaluate(X_test, y_test, verbose=0)
    deeplab_predictions = deeplab_model.predict(X_test, verbose=0)
    deeplab_iou = calculate_iou(y_test, deeplab_predictions, num_classes=3)
    
    print(f"\nDeepLab-style Test Accuracy: {deeplab_accuracy:.4f}")
    print(f"DeepLab-style Test Loss: {deeplab_loss:.4f}")
    print(f"DeepLab-style Mean IoU: {deeplab_iou:.4f}")

# 7. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

if TENSORFLOW_AVAILABLE:
    print("Image Segmentation Results:")
    print(f"1. U-Net Model:")
    print(f"   - Test Accuracy: {unet_accuracy:.4f}")
    print(f"   - Test Loss: {unet_loss:.4f}")
    print(f"   - Mean IoU: {unet_iou:.4f}")
    
    print(f"\n2. FCN Model:")
    print(f"   - Test Accuracy: {fcn_accuracy:.4f}")
    print(f"   - Test Loss: {fcn_loss:.4f}")
    print(f"   - Mean IoU: {fcn_iou:.4f}")
    
    print(f"\n3. Skip Connection Model:")
    print(f"   - Test Accuracy: {skip_accuracy:.4f}")
    print(f"   - Test Loss: {skip_loss:.4f}")
    print(f"   - Mean IoU: {skip_iou:.4f}")
    
    if 'deeplab_accuracy' in locals():
        print(f"\n4. DeepLab-style Model:")
        print(f"   - Test Accuracy: {deeplab_accuracy:.4f}")
        print(f"   - Test Loss: {deeplab_loss:.4f}")
        print(f"   - Mean IoU: {deeplab_iou:.4f}")

print(f"\nKey Insights:")
print(f"- U-Net with skip connections provides good performance for medical images")
print(f"- FCN is simpler but may lose fine details")
print(f"- Atrous convolutions help capture multi-scale features")
print(f"- IoU is a better metric than accuracy for segmentation tasks")

print(f"\nRecommendations:")
print(f"- Use U-Net for medical image segmentation")
print(f"- Use FCN for simple semantic segmentation tasks")
print(f"- Apply data augmentation to improve model robustness")
print(f"- Use appropriate evaluation metrics (IoU, Dice coefficient)")
print(f"- Consider model complexity vs performance trade-offs")
print(f"- Use skip connections to preserve fine details")
print(f"- Apply post-processing techniques (CRF, morphological operations)")
print(f"- Consider ensemble methods for improved accuracy")
print(f"- Use transfer learning with pre-trained encoders") 