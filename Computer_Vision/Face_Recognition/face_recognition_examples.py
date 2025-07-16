"""
Face Recognition Examples
=========================

- Face Detection (Haar Cascades, HOG)
- Face Feature Extraction
- Face Recognition (Eigenfaces, LBPH, Deep Learning)
- Face Embeddings
- Model evaluation and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# For deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, 
                                       Flatten, Dropout, BatchNormalization,
                                       GlobalAveragePooling2D)
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

# 1. Generate Synthetic Face Data
print("=== Face Recognition Examples ===")

def generate_synthetic_faces(n_samples=1000, img_size=64, n_persons=10):
    """Generate synthetic face-like images for recognition"""
    np.random.seed(42)
    
    images = []
    labels = []
    
    for person_id in range(n_persons):
        # Generate multiple images per person
        n_person_images = n_samples // n_persons
        
        for i in range(n_person_images):
            # Create blank image
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Add skin tone background
            skin_tone = np.random.randint(150, 220, 3)
            img[:, :] = skin_tone
            
            # Draw face outline (oval)
            center = (img_size // 2, img_size // 2)
            axes = (img_size // 3, img_size // 4)
            cv2.ellipse(img, center, axes, 0, 0, 360, (100, 100, 100), -1)
            
            # Draw eyes
            eye_y = img_size // 2 - 5
            left_eye_x = img_size // 2 - 15
            right_eye_x = img_size // 2 + 15
            
            # Random eye variations
            eye_size = np.random.randint(3, 6)
            cv2.circle(img, (left_eye_x, eye_y), eye_size, (0, 0, 0), -1)
            cv2.circle(img, (right_eye_x, eye_y), eye_size, (0, 0, 0), -1)
            
            # Draw nose
            nose_x = img_size // 2
            nose_y = img_size // 2 + 5
            cv2.circle(img, (nose_x, nose_y), 2, (100, 100, 100), -1)
            
            # Draw mouth
            mouth_y = img_size // 2 + 15
            mouth_width = np.random.randint(8, 15)
            cv2.ellipse(img, (img_size // 2, mouth_y), (mouth_width, 3), 0, 0, 180, (0, 0, 0), 2)
            
            # Add hair
            hair_y = img_size // 2 - 20
            hair_color = np.random.choice([(50, 50, 50), (100, 50, 0), (150, 100, 50)])
            cv2.ellipse(img, (img_size // 2, hair_y), (img_size // 3, img_size // 6), 0, 180, 360, hair_color, -1)
            
            # Add some facial features variations
            if np.random.random() > 0.5:
                # Add glasses
                cv2.ellipse(img, (left_eye_x, eye_y), (8, 6), 0, 0, 360, (50, 50, 50), 2)
                cv2.ellipse(img, (right_eye_x, eye_y), (8, 6), 0, 0, 360, (50, 50, 50), 2)
            
            # Add noise and variations
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = np.clip(img + noise, 0, 255)
            
            # Random brightness and contrast
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.randint(-20, 20)    # Brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            
            images.append(img)
            labels.append(person_id)
    
    return np.array(images), np.array(labels)

# Generate synthetic face data
print("Generating synthetic face data...")
X_faces, y_persons = generate_synthetic_faces()
print(f"Face data shape: {X_faces.shape}")
print(f"Number of persons: {len(np.unique(y_persons))}")
print(f"Images per person: {len(X_faces) // len(np.unique(y_persons))}")

# 2. Face Detection
print("\n=== Face Detection ===")

def detect_faces_haar(image):
    """Detect faces using Haar Cascade"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces

def detect_faces_hog(image):
    """Detect faces using HOG detector"""
    # This is a simplified version - in practice you'd use dlib
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple edge-based detection (simplified HOG)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    faces = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # Filter small contours
            faces.append([x, y, w, h])
    
    return np.array(faces)

if IMAGE_AVAILABLE:
    # Test face detection
    test_image = X_faces[0]
    
    # Haar cascade detection
    haar_faces = detect_faces_haar(test_image)
    print(f"Haar Cascade detected {len(haar_faces)} faces")
    
    # HOG detection
    hog_faces = detect_faces_hog(test_image)
    print(f"HOG detected {len(hog_faces)} faces")

# 3. Face Feature Extraction
print("\n=== Face Feature Extraction ===")

def extract_haar_features(image, feature_size=(16, 16)):
    """Extract Haar-like features from face image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, feature_size)
    
    # Simple Haar-like features (horizontal and vertical differences)
    features = []
    
    # Horizontal features
    for i in range(feature_size[0] - 1):
        for j in range(feature_size[1]):
            feature = resized[i, j] - resized[i + 1, j]
            features.append(feature)
    
    # Vertical features
    for i in range(feature_size[0]):
        for j in range(feature_size[1] - 1):
            feature = resized[i, j] - resized[i, j + 1]
            features.append(feature)
    
    return np.array(features)

def extract_lbp_features(image, radius=1, n_points=8):
    """Extract Local Binary Pattern features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simplified LBP implementation
    height, width = gray.shape
    lbp = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center = gray[i, j]
            code = 0
            
            # Sample neighbors
            for k in range(n_points):
                angle = 2 * np.pi * k / n_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                
                if gray[x, y] >= center:
                    code |= (1 << k)
            
            lbp[i, j] = code
    
    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist

def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), bins=9):
    """Extract Histogram of Oriented Gradients features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Simplified HOG calculation
    height, width = gray.shape
    cell_h, cell_w = cell_size
    n_cells_h = height // cell_h
    n_cells_w = width // cell_w
    
    hog_features = []
    
    for i in range(n_cells_h):
        for j in range(n_cells_w):
            # Extract cell
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_ori = orientation[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Calculate histogram
            hist = np.zeros(bins)
            for k in range(bins):
                angle_min = k * 180 / bins
                angle_max = (k + 1) * 180 / bins
                mask = (cell_ori >= angle_min) & (cell_ori < angle_max)
                hist[k] = np.sum(cell_mag[mask])
            
            hog_features.extend(hist)
    
    return np.array(hog_features)

# Extract features from all images
print("Extracting face features...")

haar_features = []
lbp_features = []
hog_features = []

for i, image in enumerate(X_faces):
    if i % 100 == 0:
        print(f"Processing image {i}/{len(X_faces)}")
    
    # Extract different types of features
    haar_feat = extract_haar_features(image)
    lbp_feat = extract_lbp_features(image)
    hog_feat = extract_hog_features(image)
    
    haar_features.append(haar_feat)
    lbp_features.append(lbp_feat)
    hog_features.append(hog_feat)

haar_features = np.array(haar_features)
lbp_features = np.array(lbp_features)
hog_features = np.array(hog_features)

print(f"Haar features shape: {haar_features.shape}")
print(f"LBP features shape: {lbp_features.shape}")
print(f"HOG features shape: {hog_features.shape}")

# 4. Face Recognition Models
print("\n=== Face Recognition Models ===")

# Split data for different feature types
X_train_haar, X_test_haar, y_train, y_test = train_test_split(
    haar_features, y_persons, test_size=0.2, random_state=42, stratify=y_persons
)

X_train_lbp, X_test_lbp, _, _ = train_test_split(
    lbp_features, y_persons, test_size=0.2, random_state=42, stratify=y_persons
)

X_train_hog, X_test_hog, _, _ = train_test_split(
    hog_features, y_persons, test_size=0.2, random_state=42, stratify=y_persons
)

# 4.1 Eigenfaces (PCA + SVM)
print("\n--- Eigenfaces (PCA + SVM) ---")

# Apply PCA to reduce dimensionality
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_haar)
X_test_pca = pca.transform(X_test_haar)

# Train SVM classifier
svm_classifier = SVC(kernel='rbf', probability=True)
svm_classifier.fit(X_train_pca, y_train)

# Predictions
svm_predictions = svm_classifier.predict(X_test_pca)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f"Eigenfaces + SVM Accuracy: {svm_accuracy:.4f}")

# 4.2 LBPH Face Recognition
print("\n--- LBPH Face Recognition ---")

# Train KNN classifier on LBP features
knn_lbph = KNeighborsClassifier(n_neighbors=5)
knn_lbph.fit(X_train_lbp, y_train)

# Predictions
lbph_predictions = knn_lbph.predict(X_test_lbp)
lbph_accuracy = accuracy_score(y_test, lbph_predictions)

print(f"LBPH + KNN Accuracy: {lbph_accuracy:.4f}")

# 4.3 HOG + SVM
print("\n--- HOG + SVM ---")

# Train SVM on HOG features
svm_hog = SVC(kernel='rbf', probability=True)
svm_hog.fit(X_train_hog, y_train)

# Predictions
hog_predictions = svm_hog.predict(X_test_hog)
hog_accuracy = accuracy_score(y_test, hog_predictions)

print(f"HOG + SVM Accuracy: {hog_accuracy:.4f}")

# 5. Deep Learning Face Recognition
print("\n=== Deep Learning Face Recognition ===")

def create_face_recognition_model(input_shape, num_classes):
    """Create deep learning model for face recognition"""
    
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
    # Prepare data for deep learning
    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
        X_faces, y_persons, test_size=0.2, random_state=42, stratify=y_persons
    )
    
    # Normalize images
    X_train_dl = X_train_dl.astype('float32') / 255.0
    X_test_dl = X_test_dl.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train_dl, num_classes=len(np.unique(y_persons)))
    y_test_cat = tf.keras.utils.to_categorical(y_test_dl, num_classes=len(np.unique(y_persons)))
    
    # Create and compile model
    dl_model = create_face_recognition_model(X_train_dl.shape[1:], len(np.unique(y_persons)))
    dl_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Deep Learning Model Summary:")
    dl_model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    # Train model
    print("\nTraining deep learning model...")
    dl_history = dl_model.fit(
        X_train_dl, y_train_cat,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    dl_loss, dl_accuracy = dl_model.evaluate(X_test_dl, y_test_cat, verbose=0)
    print(f"\nDeep Learning Test Accuracy: {dl_accuracy:.4f}")
    print(f"Deep Learning Test Loss: {dl_loss:.4f}")

# 6. Face Embeddings
print("\n=== Face Embeddings ===")

def create_face_embedding_model(input_shape, embedding_dim=128):
    """Create model for face embeddings"""
    
    inputs = Input(shape=input_shape)
    
    # Feature extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    
    # Embedding layer
    embeddings = Dense(embedding_dim, activation=None, name='embeddings')(x)
    
    # Normalize embeddings
    embeddings = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embeddings)
    
    model = Model(inputs=inputs, outputs=embeddings)
    return model

if TENSORFLOW_AVAILABLE:
    # Create embedding model
    embedding_model = create_face_embedding_model(X_train_dl.shape[1:], embedding_dim=128)
    embedding_model.compile(optimizer=Adam(learning_rate=0.001))
    
    print("Face Embedding Model Summary:")
    embedding_model.summary()
    
    # Generate embeddings
    print("\nGenerating face embeddings...")
    train_embeddings = embedding_model.predict(X_train_dl, verbose=0)
    test_embeddings = embedding_model.predict(X_test_dl, verbose=0)
    
    print(f"Training embeddings shape: {train_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    
    # Face recognition using embeddings
    def face_recognition_with_embeddings(train_embeddings, test_embeddings, train_labels, test_labels, threshold=0.5):
        """Perform face recognition using embeddings"""
        predictions = []
        
        for test_emb in test_embeddings:
            # Calculate distances to all training embeddings
            distances = []
            for train_emb in train_embeddings:
                distance = np.linalg.norm(test_emb - train_emb)
                distances.append(distance)
            
            # Find closest match
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            # Predict based on threshold
            if min_distance < threshold:
                predictions.append(train_labels[min_idx])
            else:
                predictions.append(-1)  # Unknown person
        
        return np.array(predictions)
    
    # Test face recognition with embeddings
    embedding_predictions = face_recognition_with_embeddings(
        train_embeddings, test_embeddings, y_train_dl, y_test_dl, threshold=0.5
    )
    
    # Calculate accuracy (excluding unknown predictions)
    known_mask = embedding_predictions != -1
    if np.sum(known_mask) > 0:
        embedding_accuracy = accuracy_score(y_test_dl[known_mask], embedding_predictions[known_mask])
        print(f"Face Recognition with Embeddings Accuracy: {embedding_accuracy:.4f}")
        print(f"Unknown faces: {np.sum(~known_mask)}/{len(embedding_predictions)}")

# 7. Model Evaluation and Visualization
print("\n=== Model Evaluation and Visualization ===")

# Compare all models
models = ['Eigenfaces+SVM', 'LBPH+KNN', 'HOG+SVM', 'Deep Learning']
accuracies = [svm_accuracy, lbph_accuracy, hog_accuracy]

if TENSORFLOW_AVAILABLE:
    accuracies.append(dl_accuracy)

# Create comparison plot
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, alpha=0.8, color=['blue', 'green', 'red', 'orange'])
plt.title('Face Recognition Model Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Visualize some face recognition results
if TENSORFLOW_AVAILABLE:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(8):
        row = i // 4
        col = i % 4
        
        # Original image
        axes[row, col].imshow(X_test_dl[i])
        
        # Get predictions from different models
        true_label = y_test_dl[i]
        
        # Deep learning prediction
        dl_pred = np.argmax(dl_model.predict(X_test_dl[i:i+1], verbose=0))
        
        # SVM prediction
        svm_pred = svm_classifier.predict(pca.transform(X_test_haar[i:i+1]))[0]
        
        title = f'True: {true_label}\nDL: {dl_pred}, SVM: {svm_pred}'
        color = 'green' if dl_pred == true_label else 'red'
        
        axes[row, col].set_title(title, color=color)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# 8. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

print("Face Recognition Results:")
print(f"1. Eigenfaces + SVM:")
print(f"   - Accuracy: {svm_accuracy:.4f}")

print(f"\n2. LBPH + KNN:")
print(f"   - Accuracy: {lbph_accuracy:.4f}")

print(f"\n3. HOG + SVM:")
print(f"   - Accuracy: {hog_accuracy:.4f}")

if TENSORFLOW_AVAILABLE:
    print(f"\n4. Deep Learning:")
    print(f"   - Accuracy: {dl_accuracy:.4f}")

print(f"\nKey Insights:")
print(f"- Deep learning models generally perform better for face recognition")
print(f"- Traditional methods (Eigenfaces, LBPH) are computationally efficient")
print(f"- Face embeddings enable flexible face recognition systems")
print(f"- Feature extraction is crucial for recognition accuracy")

print(f"\nRecommendations:")
print(f"- Use deep learning for high-accuracy face recognition")
print(f"- Apply data augmentation to improve model robustness")
print(f"- Use face embeddings for flexible recognition systems")
print(f"- Consider model complexity vs accuracy trade-offs")
print(f"- Implement proper face detection preprocessing")
print(f"- Use appropriate evaluation metrics (accuracy, precision, recall)")
print(f"- Consider ensemble methods for improved performance")
print(f"- Apply transfer learning with pre-trained face models")
print(f"- Implement proper security measures for face recognition systems") 