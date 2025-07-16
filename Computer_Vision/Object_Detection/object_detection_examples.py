"""
Object Detection Examples
=========================

- YOLO (You Only Look Once)
- R-CNN Family (R-CNN, Fast R-CNN, Faster R-CNN)
- Custom Object Detection
- Bounding Box Detection
- Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_iou, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# For deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, 
                                       Flatten, Dropout, BatchNormalization,
                                       Reshape, Concatenate, Lambda)
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

# 1. Generate Synthetic Object Detection Data
print("=== Object Detection Examples ===")

def generate_synthetic_objects(n_samples=1000, img_size=128, max_objects=3):
    """Generate synthetic images with multiple objects for detection"""
    np.random.seed(42)
    
    # Define object classes
    classes = ['circle', 'square', 'triangle']
    class_colors = {
        'circle': (255, 0, 0),    # Red
        'square': (0, 255, 0),    # Green
        'triangle': (0, 0, 255)   # Blue
    }
    
    images = []
    annotations = []
    
    for i in range(n_samples):
        # Create blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Random number of objects (1 to max_objects)
        n_objects = np.random.randint(1, max_objects + 1)
        
        img_annotations = []
        
        for j in range(n_objects):
            # Randomly select object class
            class_name = np.random.choice(classes)
            color = class_colors[class_name]
            
            # Random position and size
            x = np.random.randint(20, img_size - 40)
            y = np.random.randint(20, img_size - 40)
            size = np.random.randint(15, 30)
            
            # Draw object
            if class_name == 'circle':
                cv2.circle(img, (x, y), size, color, -1)
                bbox = [x - size, y - size, x + size, y + size]
                
            elif class_name == 'square':
                cv2.rectangle(img, (x - size, y - size), (x + size, y + size), color, -1)
                bbox = [x - size, y - size, x + size, y + size]
                
            else:  # triangle
                pts = np.array([[x, y - size], [x - size, y + size], [x + size, y + size]], np.int32)
                cv2.fillPoly(img, [pts], color)
                bbox = [x - size, y - size, x + size, y + size]
            
            # Ensure bbox is within image bounds
            bbox = np.clip(bbox, 0, img_size)
            
            # Add annotation: [class_id, x_min, y_min, x_max, y_max]
            class_id = classes.index(class_name)
            img_annotations.append([class_id] + bbox)
        
        images.append(img)
        annotations.append(img_annotations)
    
    return np.array(images), annotations

# Generate synthetic object detection data
print("Generating synthetic object detection data...")
X_images, y_annotations = generate_synthetic_objects()
print(f"Image data shape: {X_images.shape}")
print(f"Number of annotations: {sum(len(ann) for ann in y_annotations)}")

# 2. YOLO-style Object Detection
print("\n=== YOLO-style Object Detection ===")

def create_yolo_model(input_shape, num_classes, num_boxes=3):
    """Create a simplified YOLO-style model"""
    
    input_layer = Input(shape=input_shape)
    
    # Backbone network (simplified)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output layer: [x, y, w, h, confidence, class_probs...] for each box
    output_size = num_boxes * (4 + 1 + num_classes)  # 4 for bbox, 1 for confidence, num_classes for class probs
    output_layer = Dense(output_size, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def yolo_loss(y_true, y_pred, num_boxes=3, num_classes=3, lambda_coord=5.0, lambda_noobj=0.5):
    """Custom YOLO loss function"""
    
    # Reshape predictions
    batch_size = tf.shape(y_true)[0]
    y_pred = tf.reshape(y_pred, [batch_size, num_boxes, 4 + 1 + num_classes])
    
    # Extract components
    pred_boxes = y_pred[:, :, :4]
    pred_conf = y_pred[:, :, 4]
    pred_classes = y_pred[:, :, 5:]
    
    # For simplicity, we'll use a basic loss
    # In practice, you'd need to match predictions to ground truth
    box_loss = tf.reduce_mean(tf.square(pred_boxes))
    conf_loss = tf.reduce_mean(tf.square(pred_conf))
    class_loss = tf.reduce_mean(tf.square(pred_classes))
    
    total_loss = lambda_coord * box_loss + conf_loss + class_loss
    return total_loss

if TENSORFLOW_AVAILABLE:
    # Prepare data for YOLO
    # For simplicity, we'll create a simplified target format
    def prepare_yolo_targets(annotations, img_size=128, num_boxes=3, num_classes=3):
        """Prepare targets for YOLO training"""
        batch_size = len(annotations)
        targets = np.zeros((batch_size, num_boxes * (4 + 1 + num_classes)))
        
        for i, img_annotations in enumerate(annotations):
            for j, annotation in enumerate(img_annotations[:num_boxes]):
                if j < num_boxes:
                    class_id, x_min, y_min, x_max, y_max = annotation
                    
                    # Convert to center coordinates and normalize
                    x_center = (x_min + x_max) / 2 / img_size
                    y_center = (y_min + y_max) / 2 / img_size
                    width = (x_max - x_min) / img_size
                    height = (y_max - y_min) / img_size
                    
                    # Set target values
                    start_idx = j * (4 + 1 + num_classes)
                    targets[i, start_idx:start_idx+4] = [x_center, y_center, width, height]
                    targets[i, start_idx+4] = 1.0  # confidence
                    targets[i, start_idx+5+class_id] = 1.0  # class probability
        
        return targets
    
    # Prepare targets
    y_yolo = prepare_yolo_targets(y_annotations)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_images, y_yolo, test_size=0.2, random_state=42
    )
    
    # Normalize images
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Create and compile YOLO model
    yolo_model = create_yolo_model(X_train.shape[1:], num_classes=3, num_boxes=3)
    yolo_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=yolo_loss,
        metrics=['accuracy']
    )
    
    print("YOLO Model Summary:")
    yolo_model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train YOLO model
    print("\nTraining YOLO model...")
    yolo_history = yolo_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

# 3. R-CNN-style Object Detection
print("\n=== R-CNN-style Object Detection ===")

def create_rcnn_model(input_shape, num_classes):
    """Create a simplified R-CNN-style model"""
    
    input_layer = Input(shape=input_shape)
    
    # Feature extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output: class probabilities and bounding box regression
    class_output = Dense(num_classes, activation='softmax', name='class_output')(x)
    bbox_output = Dense(4, activation='sigmoid', name='bbox_output')(x)  # [x, y, w, h]
    
    model = Model(inputs=input_layer, outputs=[class_output, bbox_output])
    return model

if TENSORFLOW_AVAILABLE:
    # Prepare data for R-CNN
    def prepare_rcnn_targets(annotations, num_classes=3):
        """Prepare targets for R-CNN training"""
        batch_size = len(annotations)
        class_targets = np.zeros((batch_size, num_classes))
        bbox_targets = np.zeros((batch_size, 4))
        
        for i, img_annotations in enumerate(annotations):
            if img_annotations:
                # Use the first object for simplicity
                class_id, x_min, y_min, x_max, y_max = img_annotations[0]
                
                # Class target (one-hot encoding)
                class_targets[i, class_id] = 1.0
                
                # Bounding box target (normalized)
                bbox_targets[i] = [x_min/128, y_min/128, (x_max-x_min)/128, (y_max-y_min)/128]
        
        return class_targets, bbox_targets
    
    # Prepare targets
    y_class, y_bbox = prepare_rcnn_targets(y_annotations)
    
    # Split data
    X_train_rcnn, X_test_rcnn, y_train_class, y_test_class, y_train_bbox, y_test_bbox = train_test_split(
        X_images, y_class, y_bbox, test_size=0.2, random_state=42
    )
    
    # Normalize images
    X_train_rcnn = X_train_rcnn.astype('float32') / 255.0
    X_test_rcnn = X_test_rcnn.astype('float32') / 255.0
    
    # Create and compile R-CNN model
    rcnn_model = create_rcnn_model(X_train_rcnn.shape[1:], num_classes=3)
    rcnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'class_output': 'categorical_crossentropy',
            'bbox_output': 'mse'
        },
        loss_weights={
            'class_output': 1.0,
            'bbox_output': 1.0
        },
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'mse'
        }
    )
    
    print("R-CNN Model Summary:")
    rcnn_model.summary()
    
    # Train R-CNN model
    print("\nTraining R-CNN model...")
    rcnn_history = rcnn_model.fit(
        X_train_rcnn,
        {'class_output': y_train_class, 'bbox_output': y_train_bbox},
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

# 4. Custom Object Detection with Sliding Window
print("\n=== Custom Object Detection with Sliding Window ===")

def sliding_window_detection(image, window_size=(32, 32), stride=16):
    """Implement sliding window object detection"""
    height, width = image.shape[:2]
    detections = []
    
    for y in range(0, height - window_size[1], stride):
        for x in range(0, width - window_size[0], stride):
            # Extract window
            window = image[y:y + window_size[1], x:x + window_size[0]]
            
            # Here you would run your classifier on the window
            # For now, we'll use a simple heuristic based on color
            if np.mean(window) > 50:  # Simple threshold
                detections.append({
                    'bbox': [x, y, x + window_size[0], y + window_size[1]],
                    'confidence': np.mean(window) / 255.0
                })
    
    return detections

def non_maximum_suppression(detections, iou_threshold=0.5):
    """Apply non-maximum suppression to remove overlapping detections"""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    filtered_detections = []
    
    while detections:
        # Take the detection with highest confidence
        current = detections.pop(0)
        filtered_detections.append(current)
        
        # Remove overlapping detections
        detections = [det for det in detections if calculate_iou(current['bbox'], det['bbox']) < iou_threshold]
    
    return filtered_detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    
    # Calculate union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union

# Test sliding window detection
if IMAGE_AVAILABLE:
    test_image = X_images[0]
    detections = sliding_window_detection(test_image)
    filtered_detections = non_maximum_suppression(detections)
    
    print(f"Original detections: {len(detections)}")
    print(f"After NMS: {len(filtered_detections)}")

# 5. Model Evaluation and Visualization
print("\n=== Model Evaluation and Visualization ===")

def visualize_detections(image, detections, title="Object Detections"):
    """Visualize object detections on image"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        
        # Create rectangle
        rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                        linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add confidence text
        ax.text(bbox[0], bbox[1] - 5, f'{confidence:.2f}', 
                color='red', fontsize=10, weight='bold')
    
    ax.set_title(title)
    ax.axis('off')
    plt.show()

if TENSORFLOW_AVAILABLE and IMAGE_AVAILABLE:
    # Visualize some results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(6):
        row = i // 3
        col = i % 3
        
        # Original image with ground truth
        img = X_test[i]
        axes[row, col].imshow(img)
        
        # Add ground truth annotations
        for annotation in y_annotations[i]:
            class_id, x_min, y_min, x_max, y_max = annotation
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=2, edgecolor='green', facecolor='none')
            axes[row, col].add_patch(rect)
        
        axes[row, col].set_title(f'Image {i+1} with Ground Truth')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Model comparison
    if 'yolo_history' in locals() and 'rcnn_history' in locals():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # YOLO training history
        axes[0].plot(yolo_history.history['loss'], label='Training Loss')
        axes[0].plot(yolo_history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('YOLO Training History')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # R-CNN training history
        axes[1].plot(rcnn_history.history['loss'], label='Total Loss')
        axes[1].plot(rcnn_history.history['class_output_loss'], label='Classification Loss')
        axes[1].plot(rcnn_history.history['bbox_output_loss'], label='Bounding Box Loss')
        axes[1].set_title('R-CNN Training History')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 6. Performance Metrics
print("\n=== Performance Metrics ===")

def calculate_map(predictions, ground_truth, iou_threshold=0.5):
    """Calculate Mean Average Precision"""
    # This is a simplified implementation
    # In practice, you'd need to match predictions to ground truth
    total_predictions = sum(len(pred) for pred in predictions)
    total_ground_truth = sum(len(gt) for gt in ground_truth)
    
    if total_predictions == 0:
        return 0.0
    
    # Simplified precision calculation
    precision = total_ground_truth / total_predictions if total_predictions > 0 else 0.0
    return precision

if TENSORFLOW_AVAILABLE:
    # Evaluate models
    print("Model Evaluation:")
    
    if 'yolo_model' in locals():
        yolo_loss, yolo_acc = yolo_model.evaluate(X_test, y_test, verbose=0)
        print(f"YOLO - Loss: {yolo_loss:.4f}, Accuracy: {yolo_acc:.4f}")
    
    if 'rcnn_model' in locals():
        rcnn_results = rcnn_model.evaluate(X_test_rcnn, 
                                         {'class_output': y_test_class, 'bbox_output': y_test_bbox}, 
                                         verbose=0)
        print(f"R-CNN - Total Loss: {rcnn_results[0]:.4f}")
        print(f"R-CNN - Classification Accuracy: {rcnn_results[3]:.4f}")
        print(f"R-CNN - Bounding Box MSE: {rcnn_results[4]:.4f}")

# 7. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

print("Object Detection Results:")
if TENSORFLOW_AVAILABLE:
    if 'yolo_model' in locals():
        print(f"1. YOLO Model:")
        print(f"   - Loss: {yolo_loss:.4f}")
        print(f"   - Accuracy: {yolo_acc:.4f}")
    
    if 'rcnn_model' in locals():
        print(f"\n2. R-CNN Model:")
        print(f"   - Total Loss: {rcnn_results[0]:.4f}")
        print(f"   - Classification Accuracy: {rcnn_results[3]:.4f}")
        print(f"   - Bounding Box MSE: {rcnn_results[4]:.4f}")

print(f"\nKey Insights:")
print(f"- YOLO is faster but may be less accurate for small objects")
print(f"- R-CNN provides better accuracy but is computationally expensive")
print(f"- Sliding window approach is simple but inefficient")
print(f"- Non-maximum suppression is crucial for removing duplicate detections")

print(f"\nRecommendations:")
print(f"- Use YOLO for real-time applications")
print(f"- Use R-CNN for high-accuracy requirements")
print(f"- Apply data augmentation to improve model robustness")
print(f"- Use appropriate evaluation metrics (mAP, IoU)")
print(f"- Consider model complexity vs speed trade-offs")
print(f"- Implement proper post-processing (NMS)")
print(f"- Use transfer learning for better performance")
print(f"- Consider ensemble methods for improved accuracy") 