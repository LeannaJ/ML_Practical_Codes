"""
Feature Extraction Examples
===========================

- Traditional Feature Extraction (SIFT, SURF, ORB, HOG)
- Deep Learning Feature Extraction
- Feature Matching and Descriptors
- Feature Visualization
- Model evaluation and comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# For deep learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, 
                                       Flatten, Dropout, BatchNormalization,
                                       GlobalAveragePooling2D)
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
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
print("=== Feature Extraction Examples ===")

def generate_synthetic_images(n_samples=1000, img_size=128, n_classes=5):
    """Generate synthetic images with different patterns for feature extraction"""
    np.random.seed(42)
    
    # Define image classes
    classes = ['circles', 'lines', 'textures', 'shapes', 'patterns']
    
    images = []
    labels = []
    
    for i in range(n_samples):
        # Create blank image
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Randomly select class
        class_idx = np.random.randint(0, n_classes)
        class_name = classes[class_idx]
        
        # Generate different patterns based on class
        if class_name == 'circles':
            # Draw multiple circles
            for _ in range(np.random.randint(3, 8)):
                center = (np.random.randint(20, img_size-20), np.random.randint(20, img_size-20))
                radius = np.random.randint(10, 25)
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                cv2.circle(img, center, radius, color, -1)
                
        elif class_name == 'lines':
            # Draw lines
            for _ in range(np.random.randint(5, 15)):
                pt1 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
                pt2 = (np.random.randint(0, img_size), np.random.randint(0, img_size))
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                thickness = np.random.randint(2, 6)
                cv2.line(img, pt1, pt2, color, thickness)
                
        elif class_name == 'textures':
            # Create texture patterns
            for x in range(0, img_size, 10):
                for y in range(0, img_size, 10):
                    if np.random.random() > 0.5:
                        color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                        cv2.rectangle(img, (x, y), (x+8, y+8), color, -1)
                        
        elif class_name == 'shapes':
            # Draw various shapes
            for _ in range(np.random.randint(2, 6)):
                shape_type = np.random.choice(['rectangle', 'triangle', 'polygon'])
                color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                
                if shape_type == 'rectangle':
                    x = np.random.randint(10, img_size-30)
                    y = np.random.randint(10, img_size-30)
                    w = np.random.randint(20, 40)
                    h = np.random.randint(20, 40)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
                    
                elif shape_type == 'triangle':
                    pts = np.array([[np.random.randint(10, img_size-10), np.random.randint(10, img_size//2)],
                                   [np.random.randint(10, img_size//2), np.random.randint(img_size//2, img_size-10)],
                                   [np.random.randint(img_size//2, img_size-10), np.random.randint(img_size//2, img_size-10)]], np.int32)
                    cv2.fillPoly(img, [pts], color)
                    
                else:  # polygon
                    pts = np.array([[np.random.randint(10, img_size-10) for _ in range(6)] for _ in range(2)]).T
                    cv2.fillPoly(img, [pts], color)
                    
        else:  # patterns
            # Create geometric patterns
            for i in range(0, img_size, 20):
                for j in range(0, img_size, 20):
                    if (i + j) % 40 == 0:
                        color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                        cv2.circle(img, (i, j), 5, color, -1)
                    else:
                        color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                        cv2.rectangle(img, (i, j), (i+10, j+10), color, -1)
        
        # Add noise
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = np.clip(img + noise, 0, 255)
        
        images.append(img)
        labels.append(class_idx)
    
    return np.array(images), np.array(labels)

# Generate synthetic image data
print("Generating synthetic image data...")
X_images, y_labels = generate_synthetic_images()
print(f"Image data shape: {X_images.shape}")
print(f"Number of classes: {len(np.unique(y_labels))}")
print(f"Classes: {np.unique(y_labels)}")

# 2. Traditional Feature Extraction
print("\n=== Traditional Feature Extraction ===")

def extract_sift_features(image, max_features=100):
    """Extract SIFT features from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    if descriptors is None:
        # Return zero vector if no features found
        return np.zeros(128)
    
    # Return mean descriptor
    return np.mean(descriptors, axis=0)

def extract_orb_features(image, max_features=100):
    """Extract ORB features from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    if descriptors is None:
        # Return zero vector if no features found
        return np.zeros(32)
    
    # Return mean descriptor
    return np.mean(descriptors, axis=0)

def extract_hog_features(image, cell_size=(8, 8), block_size=(2, 2), bins=9):
    """Extract Histogram of Oriented Gradients features"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # Calculate HOG features
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

def extract_color_histogram(image, bins=32):
    """Extract color histogram features"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Calculate histograms for each channel
    histograms = []
    
    # BGR histograms
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    
    # HSV histograms
    for i in range(3):
        hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    
    # LAB histograms
    for i in range(3):
        hist = cv2.calcHist([lab], [i], None, [bins], [0, 256])
        histograms.extend(hist.flatten())
    
    return np.array(histograms)

# Extract traditional features
print("Extracting traditional features...")

sift_features = []
orb_features = []
hog_features = []
lbp_features = []
color_features = []

for i, image in enumerate(X_images):
    if i % 100 == 0:
        print(f"Processing image {i}/{len(X_images)}")
    
    # Extract different types of features
    sift_feat = extract_sift_features(image)
    orb_feat = extract_orb_features(image)
    hog_feat = extract_hog_features(image)
    lbp_feat = extract_lbp_features(image)
    color_feat = extract_color_histogram(image)
    
    sift_features.append(sift_feat)
    orb_features.append(orb_feat)
    hog_features.append(hog_feat)
    lbp_features.append(lbp_feat)
    color_features.append(color_feat)

sift_features = np.array(sift_features)
orb_features = np.array(orb_features)
hog_features = np.array(hog_features)
lbp_features = np.array(lbp_features)
color_features = np.array(color_features)

print(f"SIFT features shape: {sift_features.shape}")
print(f"ORB features shape: {orb_features.shape}")
print(f"HOG features shape: {hog_features.shape}")
print(f"LBP features shape: {lbp_features.shape}")
print(f"Color features shape: {color_features.shape}")

# 3. Deep Learning Feature Extraction
print("\n=== Deep Learning Feature Extraction ===")

def create_feature_extraction_model(input_shape, feature_dim=512):
    """Create a custom feature extraction model"""
    
    inputs = Input(shape=input_shape)
    
    # Feature extraction layers
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
    
    # Feature vector
    features = Dense(feature_dim, activation='relu', name='features')(x)
    
    model = Model(inputs=inputs, outputs=features)
    return model

def extract_pretrained_features(model_name, images):
    """Extract features using pre-trained models"""
    
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create feature extraction model
    feature_model = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Extract features
    features = feature_model.predict(images, verbose=0)
    
    # Global average pooling
    features = GlobalAveragePooling2D()(features)
    
    return features

if TENSORFLOW_AVAILABLE:
    # Prepare data for deep learning
    X_dl = X_images.astype('float32') / 255.0
    
    # Create custom feature extraction model
    custom_model = create_feature_extraction_model(X_dl.shape[1:], feature_dim=512)
    custom_model.compile(optimizer=Adam(learning_rate=0.001))
    
    print("Custom Feature Extraction Model Summary:")
    custom_model.summary()
    
    # Extract custom features
    print("\nExtracting custom deep learning features...")
    custom_features = custom_model.predict(X_dl, verbose=0)
    print(f"Custom features shape: {custom_features.shape}")
    
    # Extract pre-trained features
    print("\nExtracting pre-trained features...")
    
    try:
        vgg_features = extract_pretrained_features('VGG16', X_dl)
        print(f"VGG16 features shape: {vgg_features.shape}")
    except Exception as e:
        print(f"Error extracting VGG16 features: {e}")
        vgg_features = None
    
    try:
        resnet_features = extract_pretrained_features('ResNet50', X_dl)
        print(f"ResNet50 features shape: {resnet_features.shape}")
    except Exception as e:
        print(f"Error extracting ResNet50 features: {e}")
        resnet_features = None

# 4. Feature Analysis and Comparison
print("\n=== Feature Analysis and Comparison ===")

def evaluate_features(features, labels, feature_name):
    """Evaluate features using SVM classifier"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train SVM classifier
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    
    # Predictions
    predictions = svm.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"{feature_name} - Accuracy: {accuracy:.4f}")
    return accuracy

# Evaluate all feature types
print("Evaluating feature extraction methods...")

feature_results = {}

# Traditional features
feature_results['SIFT'] = evaluate_features(sift_features, y_labels, 'SIFT')
feature_results['ORB'] = evaluate_features(orb_features, y_labels, 'ORB')
feature_results['HOG'] = evaluate_features(hog_features, y_labels, 'HOG')
feature_results['LBP'] = evaluate_features(lbp_features, y_labels, 'LBP')
feature_results['Color'] = evaluate_features(color_features, y_labels, 'Color')

# Deep learning features
if TENSORFLOW_AVAILABLE:
    feature_results['Custom'] = evaluate_features(custom_features, y_labels, 'Custom')
    
    if vgg_features is not None:
        feature_results['VGG16'] = evaluate_features(vgg_features, y_labels, 'VGG16')
    
    if resnet_features is not None:
        feature_results['ResNet50'] = evaluate_features(resnet_features, y_labels, 'ResNet50')

# 5. Feature Visualization
print("\n=== Feature Visualization ===")

def visualize_features_2d(features, labels, title, method='PCA'):
    """Visualize features in 2D using PCA or t-SNE"""
    
    if method == 'PCA':
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42)
    
    # Reduce dimensionality
    features_2d = reducer.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f'{title} - {method}')
    plt.xlabel(f'{method} Component 1')
    plt.ylabel(f'{method} Component 2')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.show()

# Visualize different feature types
print("Visualizing feature distributions...")

# SIFT features
visualize_features_2d(sift_features, y_labels, 'SIFT Features', 'PCA')

# HOG features
visualize_features_2d(hog_features, y_labels, 'HOG Features', 'PCA')

# Custom deep learning features
if TENSORFLOW_AVAILABLE:
    visualize_features_2d(custom_features, y_labels, 'Custom Deep Features', 'PCA')

# 6. Feature Matching and Descriptors
print("\n=== Feature Matching and Descriptors ===")

def feature_matching_demo(image1, image2):
    """Demonstrate feature matching between two images"""
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        print("No features detected in one or both images")
        return
    
    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"Total matches: {len(matches)}")
    print(f"Good matches: {len(good_matches)}")
    
    # Draw matches
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching Results')
    plt.axis('off')
    plt.show()

if IMAGE_AVAILABLE:
    # Test feature matching
    print("Testing feature matching...")
    feature_matching_demo(X_images[0], X_images[100])

# 7. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

def analyze_feature_importance(features, labels, feature_name):
    """Analyze feature importance using Random Forest"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get feature importance
    importance = rf.feature_importances_
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance)
    plt.title(f'{feature_name} - Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print top features
    top_indices = np.argsort(importance)[-10:]
    print(f"\nTop 10 important features for {feature_name}:")
    for i, idx in enumerate(reversed(top_indices)):
        print(f"  {i+1}. Feature {idx}: {importance[idx]:.4f}")

# Analyze feature importance for different methods
print("Analyzing feature importance...")

analyze_feature_importance(hog_features, y_labels, 'HOG')
analyze_feature_importance(lbp_features, y_labels, 'LBP')

if TENSORFLOW_AVAILABLE:
    analyze_feature_importance(custom_features, y_labels, 'Custom Deep Features')

# 8. Model Comparison and Summary
print("\n=== Model Comparison and Summary ===")

# Create comparison plot
methods = list(feature_results.keys())
accuracies = list(feature_results.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(methods, accuracies, alpha=0.8, color=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray'])
plt.title('Feature Extraction Method Comparison')
plt.ylabel('Classification Accuracy')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Summary
print("\nFeature Extraction Results:")
for method, accuracy in feature_results.items():
    print(f"{method}: {accuracy:.4f}")

print(f"\nKey Insights:")
print(f"- Deep learning features generally perform better than traditional methods")
print(f"- Pre-trained models provide strong feature representations")
print(f"- Traditional methods are computationally efficient")
print(f"- Feature selection can improve model performance")

print(f"\nRecommendations:")
print(f"- Use deep learning features for high-accuracy applications")
print(f"- Apply traditional methods for real-time or resource-constrained scenarios")
print(f"- Consider ensemble methods combining multiple feature types")
print(f"- Use feature selection to reduce dimensionality")
print(f"- Apply appropriate preprocessing for each feature type")
print(f"- Consider domain-specific feature extraction methods")
print(f"- Use transfer learning for feature extraction")
print(f"- Implement proper feature normalization and scaling") 