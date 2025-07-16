"""
Clustering Examples
==================

- Customer Segmentation (K-means, Hierarchical, DBSCAN)
- Image Segmentation (K-means, Mean Shift)
- Document Clustering (TF-IDF + K-means)
- Anomaly Detection (Isolation Forest, Local Outlier Factor)
- Model comparison and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN, 
                           MeanShift, SpectralClustering, OPTICS)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                           davies_bouldin_score, adjusted_rand_score)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# For image processing
try:
    from PIL import Image
    import cv2
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False
    print("OpenCV/PIL not available. Install with: pip install opencv-python pillow")

# 1. Customer Segmentation
print("=== Customer Segmentation ===")

def generate_customer_data(n_customers=2000):
    """Generate synthetic customer data for segmentation"""
    np.random.seed(42)
    
    # Generate customer features
    age = np.random.normal(35, 12, n_customers)
    age = np.clip(age, 18, 80)
    
    income = np.random.exponential(scale=50000, size=n_customers)
    income = np.clip(income, 20000, 200000)
    
    spending_score = np.random.uniform(1, 100, n_customers)
    
    # Generate different customer segments
    n_segments = 4
    segment_sizes = [n_customers // n_segments] * n_segments
    segment_sizes[0] += n_customers % n_segments  # Add remainder to first segment
    
    # Segment 1: High income, high spending (Premium customers)
    start_idx = 0
    end_idx = segment_sizes[0]
    age[start_idx:end_idx] = np.random.normal(45, 8, segment_sizes[0])
    income[start_idx:end_idx] = np.random.exponential(scale=80000, size=segment_sizes[0])
    spending_score[start_idx:end_idx] = np.random.uniform(70, 100, segment_sizes[0])
    
    # Segment 2: High income, low spending (Conservative customers)
    start_idx = end_idx
    end_idx += segment_sizes[1]
    age[start_idx:end_idx] = np.random.normal(50, 10, segment_sizes[1])
    income[start_idx:end_idx] = np.random.exponential(scale=70000, size=segment_sizes[1])
    spending_score[start_idx:end_idx] = np.random.uniform(1, 40, segment_sizes[1])
    
    # Segment 3: Low income, high spending (Young spenders)
    start_idx = end_idx
    end_idx += segment_sizes[2]
    age[start_idx:end_idx] = np.random.normal(25, 5, segment_sizes[2])
    income[start_idx:end_idx] = np.random.exponential(scale=30000, size=segment_sizes[2])
    spending_score[start_idx:end_idx] = np.random.uniform(60, 90, segment_sizes[2])
    
    # Segment 4: Low income, low spending (Budget customers)
    start_idx = end_idx
    end_idx += segment_sizes[3]
    age[start_idx:end_idx] = np.random.normal(30, 8, segment_sizes[3])
    income[start_idx:end_idx] = np.random.exponential(scale=25000, size=segment_sizes[3])
    spending_score[start_idx:end_idx] = np.random.uniform(1, 30, segment_sizes[3])
    
    # Generate additional features
    gender = np.random.choice(['Male', 'Female'], n_customers)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers, p=[0.3, 0.4, 0.2, 0.1])
    
    # Create true labels for evaluation
    true_labels = []
    for i, size in enumerate(segment_sizes):
        true_labels.extend([i] * size)
    
    return pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': age,
        'income': income,
        'spending_score': spending_score,
        'gender': gender,
        'education': education,
        'true_segment': true_labels
    })

# Generate customer data
customer_data = generate_customer_data()
print(f"Customer data shape: {customer_data.shape}")
print(f"True segments: {np.unique(customer_data['true_segment'], return_counts=True)}")

# Prepare features for clustering
customer_features = ['age', 'income', 'spending_score']
X_customer = customer_data[customer_features].values
y_customer_true = customer_data['true_segment'].values

# Scale features
scaler_customer = StandardScaler()
X_customer_scaled = scaler_customer.fit_transform(X_customer)

# 2. Image Segmentation
print("\n=== Image Segmentation ===")

def generate_synthetic_image(n_pixels=10000):
    """Generate synthetic image data for segmentation"""
    np.random.seed(42)
    
    # Create a synthetic image with 3 segments
    n_segments = 3
    segment_sizes = [n_pixels // n_segments] * n_segments
    segment_sizes[0] += n_pixels % n_segments
    
    # Generate RGB values for each segment
    image_data = []
    true_segments = []
    
    # Segment 1: Red dominant
    start_idx = 0
    end_idx = segment_sizes[0]
    r_values = np.random.normal(200, 30, segment_sizes[0])
    g_values = np.random.normal(50, 20, segment_sizes[0])
    b_values = np.random.normal(50, 20, segment_sizes[0])
    
    for i in range(start_idx, end_idx):
        image_data.append([r_values[i-start_idx], g_values[i-start_idx], b_values[i-start_idx]])
        true_segments.append(0)
    
    # Segment 2: Green dominant
    start_idx = end_idx
    end_idx += segment_sizes[1]
    r_values = np.random.normal(50, 20, segment_sizes[1])
    g_values = np.random.normal(200, 30, segment_sizes[1])
    b_values = np.random.normal(50, 20, segment_sizes[1])
    
    for i in range(start_idx, end_idx):
        image_data.append([r_values[i-start_idx], g_values[i-start_idx], b_values[i-start_idx]])
        true_segments.append(1)
    
    # Segment 3: Blue dominant
    start_idx = end_idx
    end_idx += segment_sizes[2]
    r_values = np.random.normal(50, 20, segment_sizes[2])
    g_values = np.random.normal(50, 20, segment_sizes[2])
    b_values = np.random.normal(200, 30, segment_sizes[2])
    
    for i in range(start_idx, end_idx):
        image_data.append([r_values[i-start_idx], g_values[i-start_idx], b_values[i-start_idx]])
        true_segments.append(2)
    
    return np.array(image_data), np.array(true_segments)

# Generate synthetic image data
image_data, image_true_segments = generate_synthetic_image()
print(f"Image data shape: {image_data.shape}")
print(f"Image segments: {np.unique(image_true_segments, return_counts=True)}")

# Scale image data
scaler_image = StandardScaler()
image_data_scaled = scaler_image.fit_transform(image_data)

# 3. Document Clustering
print("\n=== Document Clustering ===")

def generate_document_data(n_documents=500):
    """Generate synthetic document data for clustering"""
    np.random.seed(42)
    
    # Define topics and their keywords
    topics = {
        'Technology': ['artificial intelligence', 'machine learning', 'blockchain', 'cybersecurity', 'cloud computing', 'data science', 'programming', 'software'],
        'Business': ['marketing', 'finance', 'strategy', 'management', 'entrepreneurship', 'investment', 'startup', 'corporate'],
        'Health': ['medical', 'healthcare', 'fitness', 'nutrition', 'wellness', 'medicine', 'therapy', 'disease'],
        'Education': ['learning', 'teaching', 'academic', 'research', 'university', 'student', 'knowledge', 'study']
    }
    
    documents = []
    true_topics = []
    
    for topic_name, keywords in topics.items():
        n_docs_per_topic = n_documents // len(topics)
        
        for _ in range(n_docs_per_topic):
            # Generate document text based on topic keywords
            doc_length = np.random.randint(50, 200)
            doc_words = []
            
            # Add topic-specific keywords
            n_topic_words = np.random.randint(5, 15)
            topic_words = np.random.choice(keywords, n_topic_words, replace=True)
            doc_words.extend(topic_words)
            
            # Add some general words
            general_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            n_general_words = doc_length - len(doc_words)
            general_words_sample = np.random.choice(general_words, n_general_words, replace=True)
            doc_words.extend(general_words_sample)
            
            # Shuffle words
            np.random.shuffle(doc_words)
            
            documents.append(' '.join(doc_words))
            true_topics.append(topic_name)
    
    return documents, true_topics

# Generate document data
documents, document_true_topics = generate_document_data()
print(f"Number of documents: {len(documents)}")
print(f"Document topics: {np.unique(document_true_topics, return_counts=True)}")

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
document_features = tfidf.fit_transform(documents)

# 4. Clustering Algorithms
print("\n=== Clustering Algorithms ===")

def apply_clustering_algorithms(X, true_labels=None, dataset_name=""):
    """Apply various clustering algorithms and evaluate performance"""
    algorithms = {
        'K-means': KMeans(n_clusters=4, random_state=42),
        'Hierarchical': AgglomerativeClustering(n_clusters=4),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Mean Shift': MeanShift(bandwidth=2),
        'Spectral': SpectralClustering(n_clusters=4, random_state=42)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"\nTraining {name} for {dataset_name}...")
        
        # Fit and predict
        if name == 'DBSCAN':
            # DBSCAN might not find exactly 4 clusters
            labels = algorithm.fit_predict(X)
        else:
            labels = algorithm.fit_predict(X)
        
        # Calculate metrics
        n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise points for DBSCAN
        
        # Silhouette score (only if more than 1 cluster)
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = 0
        
        # Calinski-Harabasz score
        if n_clusters > 1:
            calinski = calinski_harabasz_score(X, labels)
        else:
            calinski = 0
        
        # Davies-Bouldin score
        if n_clusters > 1:
            davies = davies_bouldin_score(X, labels)
        else:
            davies = float('inf')
        
        results[name] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies
        }
        
        # Calculate ARI if true labels are available
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            results[name]['adjusted_rand_score'] = ari
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Silhouette score: {silhouette:.3f}")
        if true_labels is not None:
            print(f"  Adjusted Rand Index: {ari:.3f}")
    
    return results

# Apply clustering to customer data
customer_results = apply_clustering_algorithms(X_customer_scaled, y_customer_true, "Customer Segmentation")

# Apply clustering to image data
image_results = apply_clustering_algorithms(image_data_scaled, image_true_segments, "Image Segmentation")

# Apply clustering to document data
document_results = apply_clustering_algorithms(document_features.toarray(), None, "Document Clustering")

# 5. Optimal Number of Clusters
print("\n=== Optimal Number of Clusters ===")

def find_optimal_clusters(X, max_clusters=10):
    """Find optimal number of clusters using various methods"""
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
        davies_scores.append(davies_bouldin_score(X, labels))
    
    return silhouette_scores, calinski_scores, davies_scores

# Find optimal clusters for customer data
silhouette_scores, calinski_scores, davies_scores = find_optimal_clusters(X_customer_scaled)

# Plot elbow curves
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

n_clusters_range = range(2, 11)

axes[0].plot(n_clusters_range, silhouette_scores, 'bo-')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score vs Number of Clusters')
axes[0].grid(True, alpha=0.3)

axes[1].plot(n_clusters_range, calinski_scores, 'ro-')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Calinski-Harabasz Score')
axes[1].set_title('Calinski-Harabasz Score vs Number of Clusters')
axes[1].grid(True, alpha=0.3)

axes[2].plot(n_clusters_range, davies_scores, 'go-')
axes[2].set_xlabel('Number of Clusters')
axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('Davies-Bouldin Score vs Number of Clusters')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Hierarchical Clustering Dendrogram
print("\n=== Hierarchical Clustering Dendrogram ===")

# Create dendrogram for customer data (using a sample for visualization)
sample_size = min(100, len(X_customer_scaled))
sample_indices = np.random.choice(len(X_customer_scaled), sample_size, replace=False)
X_sample = X_customer_scaled[sample_indices]

# Create linkage matrix
linkage_matrix = linkage(X_sample, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=sample_indices, leaf_rotation=90, leaf_font_size=8)
plt.title('Hierarchical Clustering Dendrogram (Customer Data)')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.show()

# 7. Dimensionality Reduction and Visualization
print("\n=== Dimensionality Reduction and Visualization ===")

# Apply PCA for visualization
pca = PCA(n_components=2)
X_customer_pca = pca.fit_transform(X_customer_scaled)

# Apply t-SNE for better visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_customer_tsne = tsne.fit_transform(X_customer_scaled)

# Plot clustering results
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# True labels
scatter = axes[0, 0].scatter(X_customer_pca[:, 0], X_customer_pca[:, 1], c=y_customer_true, cmap='viridis', alpha=0.6)
axes[0, 0].set_title('True Customer Segments (PCA)')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[0, 0])

# K-means results
kmeans_labels = customer_results['K-means']['labels']
scatter = axes[0, 1].scatter(X_customer_pca[:, 0], X_customer_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
axes[0, 1].set_title('K-means Clustering (PCA)')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[0, 1])

# Hierarchical results
hierarchical_labels = customer_results['Hierarchical']['labels']
scatter = axes[0, 2].scatter(X_customer_pca[:, 0], X_customer_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
axes[0, 2].set_title('Hierarchical Clustering (PCA)')
axes[0, 2].set_xlabel('PC1')
axes[0, 2].set_ylabel('PC2')
plt.colorbar(scatter, ax=axes[0, 2])

# t-SNE visualization
scatter = axes[1, 0].scatter(X_customer_tsne[:, 0], X_customer_tsne[:, 1], c=y_customer_true, cmap='viridis', alpha=0.6)
axes[1, 0].set_title('True Customer Segments (t-SNE)')
axes[1, 0].set_xlabel('t-SNE 1')
axes[1, 0].set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=axes[1, 0])

scatter = axes[1, 1].scatter(X_customer_tsne[:, 0], X_customer_tsne[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
axes[1, 1].set_title('K-means Clustering (t-SNE)')
axes[1, 1].set_xlabel('t-SNE 1')
axes[1, 1].set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=axes[1, 1])

scatter = axes[1, 2].scatter(X_customer_tsne[:, 0], X_customer_tsne[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
axes[1, 2].set_title('Hierarchical Clustering (t-SNE)')
axes[1, 2].set_xlabel('t-SNE 1')
axes[1, 2].set_ylabel('t-SNE 2')
plt.colorbar(scatter, ax=axes[1, 2])

plt.tight_layout()
plt.show()

# 8. Anomaly Detection using Clustering
print("\n=== Anomaly Detection using Clustering ===")

def detect_anomalies_clustering(X, contamination=0.1):
    """Detect anomalies using clustering-based methods"""
    methods = {
        'Isolation Forest': IsolationForest(contamination=contamination, random_state=42),
        'Local Outlier Factor': LocalOutlierFactor(contamination=contamination, novelty=False),
        'DBSCAN (Outliers)': DBSCAN(eps=0.5, min_samples=5)
    }
    
    results = {}
    
    for name, method in methods.items():
        print(f"\nDetecting anomalies using {name}...")
        
        if name == 'Local Outlier Factor':
            # LOF doesn't have predict method, use fit_predict
            labels = method.fit_predict(X)
            # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
            anomalies = (labels == -1).astype(int)
        elif name == 'DBSCAN (Outliers)':
            labels = method.fit_predict(X)
            # DBSCAN: -1 (noise) -> 1 (anomaly), others -> 0 (normal)
            anomalies = (labels == -1).astype(int)
        else:
            # Isolation Forest
            method.fit(X)
            labels = method.predict(X)
            # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
            anomalies = (labels == -1).astype(int)
        
        n_anomalies = np.sum(anomalies)
        anomaly_rate = n_anomalies / len(anomalies)
        
        results[name] = {
            'anomalies': anomalies,
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate
        }
        
        print(f"  Anomalies detected: {n_anomalies}")
        print(f"  Anomaly rate: {anomaly_rate:.3f}")
    
    return results

# Detect anomalies in customer data
anomaly_results = detect_anomalies_clustering(X_customer_scaled)

# 9. Model Comparison
print("\n=== Model Comparison ===")

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Compare silhouette scores
algorithms = list(customer_results.keys())
silhouette_scores = [customer_results[alg]['silhouette_score'] for alg in algorithms]

bars = axes[0, 0].bar(algorithms, silhouette_scores, alpha=0.8)
axes[0, 0].set_xlabel('Algorithms')
axes[0, 0].set_ylabel('Silhouette Score')
axes[0, 0].set_title('Silhouette Score Comparison')
axes[0, 0].set_xticklabels(algorithms, rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, silhouette_scores):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

# Compare Calinski-Harabasz scores
calinski_scores = [customer_results[alg]['calinski_harabasz_score'] for alg in algorithms]

bars = axes[0, 1].bar(algorithms, calinski_scores, alpha=0.8, color='orange')
axes[0, 1].set_xlabel('Algorithms')
axes[0, 1].set_ylabel('Calinski-Harabasz Score')
axes[0, 1].set_title('Calinski-Harabasz Score Comparison')
axes[0, 1].set_xticklabels(algorithms, rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, calinski_scores):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.0f}', ha='center', va='bottom')

# Compare Davies-Bouldin scores (lower is better)
davies_scores = [customer_results[alg]['davies_bouldin_score'] for alg in algorithms]

bars = axes[1, 0].bar(algorithms, davies_scores, alpha=0.8, color='green')
axes[1, 0].set_xlabel('Algorithms')
axes[1, 0].set_ylabel('Davies-Bouldin Score')
axes[1, 0].set_title('Davies-Bouldin Score Comparison (Lower is Better)')
axes[1, 0].set_xticklabels(algorithms, rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, davies_scores):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

# Compare Adjusted Rand Index (if available)
if 'adjusted_rand_score' in customer_results[algorithms[0]]:
    ari_scores = [customer_results[alg]['adjusted_rand_score'] for alg in algorithms]
    
    bars = axes[1, 1].bar(algorithms, ari_scores, alpha=0.8, color='red')
    axes[1, 1].set_xlabel('Algorithms')
    axes[1, 1].set_ylabel('Adjusted Rand Index')
    axes[1, 1].set_title('Adjusted Rand Index Comparison')
    axes[1, 1].set_xticklabels(algorithms, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, ari_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
else:
    # Plot number of clusters found
    n_clusters = [customer_results[alg]['n_clusters'] for alg in algorithms]
    
    bars = axes[1, 1].bar(algorithms, n_clusters, alpha=0.8, color='purple')
    axes[1, 1].set_xlabel('Algorithms')
    axes[1, 1].set_ylabel('Number of Clusters')
    axes[1, 1].set_title('Number of Clusters Found')
    axes[1, 1].set_xticklabels(algorithms, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, n_cluster in zip(bars, n_clusters):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{n_cluster}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 10. Customer Segment Analysis
print("\n=== Customer Segment Analysis ===")

# Analyze K-means results
kmeans_labels = customer_results['K-means']['labels']
customer_data['kmeans_segment'] = kmeans_labels

# Segment analysis
segment_analysis = customer_data.groupby('kmeans_segment').agg({
    'age': ['mean', 'std'],
    'income': ['mean', 'std'],
    'spending_score': ['mean', 'std'],
    'gender': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
    'education': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
}).round(2)

print("Customer Segment Analysis (K-means):")
print(segment_analysis)

# Visualize segment characteristics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Age distribution by segment
for segment in np.unique(kmeans_labels):
    segment_data = customer_data[customer_data['kmeans_segment'] == segment]
    axes[0, 0].hist(segment_data['age'], alpha=0.7, label=f'Segment {segment}', bins=20)

axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Age Distribution by Segment')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Income vs Spending Score by segment
for segment in np.unique(kmeans_labels):
    segment_data = customer_data[customer_data['kmeans_segment'] == segment]
    axes[0, 1].scatter(segment_data['income'], segment_data['spending_score'], 
                      alpha=0.6, label=f'Segment {segment}')

axes[0, 1].set_xlabel('Income')
axes[0, 1].set_ylabel('Spending Score')
axes[0, 1].set_title('Income vs Spending Score by Segment')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gender distribution by segment
gender_counts = customer_data.groupby(['kmeans_segment', 'gender']).size().unstack(fill_value=0)
gender_counts.plot(kind='bar', ax=axes[1, 0], alpha=0.8)
axes[1, 0].set_xlabel('Segment')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Gender Distribution by Segment')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Education distribution by segment
education_counts = customer_data.groupby(['kmeans_segment', 'education']).size().unstack(fill_value=0)
education_counts.plot(kind='bar', ax=axes[1, 1], alpha=0.8)
axes[1, 1].set_xlabel('Segment')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Education Distribution by Segment')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 11. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

# Find best algorithm for customer segmentation
def find_best_algorithm(results, metric='silhouette_score'):
    best_algorithm = max(results.items(), key=lambda x: x[1][metric])
    return best_algorithm[0], best_algorithm[1]

print("Best Clustering Algorithm for Customer Segmentation:")
best_alg, best_metrics = find_best_algorithm(customer_results)
print(f"Algorithm: {best_alg}")
print(f"Silhouette Score: {best_metrics['silhouette_score']:.3f}")
print(f"Calinski-Harabasz Score: {best_metrics['calinski_harabasz_score']:.0f}")
print(f"Davies-Bouldin Score: {best_metrics['davies_bouldin_score']:.3f}")

if 'adjusted_rand_score' in best_metrics:
    print(f"Adjusted Rand Index: {best_metrics['adjusted_rand_score']:.3f}")

print(f"\nOptimal Number of Clusters:")
optimal_silhouette = np.argmax(silhouette_scores) + 2
optimal_calinski = np.argmax(calinski_scores) + 2
optimal_davies = np.argmin(davies_scores) + 2

print(f"Based on Silhouette Score: {optimal_silhouette}")
print(f"Based on Calinski-Harabasz Score: {optimal_calinski}")
print(f"Based on Davies-Bouldin Score: {optimal_davies}")

print(f"\nAnomaly Detection Results:")
for method, result in anomaly_results.items():
    print(f"{method}: {result['n_anomalies']} anomalies ({result['anomaly_rate']:.1%})")

print(f"\nKey Insights:")
print(f"- K-means performs well for well-separated clusters")
print(f"- Hierarchical clustering provides interpretable dendrograms")
print(f"- DBSCAN is effective for irregular cluster shapes")
print(f"- Silhouette score is a good metric for cluster quality")
print(f"- Feature scaling is crucial for clustering algorithms")

print(f"\nRecommendations:")
print(f"- Use K-means for large datasets with spherical clusters")
print(f"- Use Hierarchical clustering for small datasets and interpretability")
print(f"- Use DBSCAN for irregular cluster shapes and noise detection")
print(f"- Always scale features before clustering")
print(f"- Use multiple evaluation metrics for comprehensive assessment")
print(f"- Consider domain knowledge when choosing number of clusters")
print(f"- Use dimensionality reduction for visualization")
print(f"- Combine clustering with anomaly detection for robust analysis") 