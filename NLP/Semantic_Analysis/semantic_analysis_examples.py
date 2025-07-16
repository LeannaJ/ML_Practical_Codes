"""
Semantic Analysis Examples
=========================

- Sentence Embedding (SBERT, Universal Sentence Encoder)
- Semantic similarity (cosine similarity)
- Semantic clustering (KMeans)
- Paraphrase detection
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. SBERT (Sentence-BERT)
print("\n=== Sentence Embedding with SBERT ===")
try:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "AI includes machine learning as one of its components.",
        "The weather forecast predicts rain tomorrow.",
        "Tomorrow's weather will be rainy according to the forecast.",
        "Deep learning models use neural networks for pattern recognition.",
        "Neural networks are the foundation of deep learning systems.",
        "Natural language processing enables computers to understand text.",
        "Computers can comprehend human language through NLP techniques.",
        "Data science combines statistics and programming for insights.",
        "Programming and statistics are essential skills in data science."
    ]
    embeddings = sbert.encode(sentences)
    print("SBERT embedding shape:", embeddings.shape)
except Exception as e:
    print("SBERT model loading failed. (pip install sentence-transformers required)")
    print("Error:", e)
    # Fallback to random embeddings for demonstration
    embeddings = np.random.randn(10, 384)
    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "AI includes machine learning as one of its components.",
        "The weather forecast predicts rain tomorrow.",
        "Tomorrow's weather will be rainy according to the forecast.",
        "Deep learning models use neural networks for pattern recognition.",
        "Neural networks are the foundation of deep learning systems.",
        "Natural language processing enables computers to understand text.",
        "Computers can comprehend human language through NLP techniques.",
        "Data science combines statistics and programming for insights.",
        "Programming and statistics are essential skills in data science."
    ]

# 2. Semantic similarity analysis
print("\n=== Semantic Similarity Analysis ===")
print("Pairwise similarity matrix:")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        print(f"Sentences {i+1} & {j+1}: {sim:.4f}")

# 3. Semantic clustering (KMeans)
print("\n=== Semantic Clustering (KMeans) ===")
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(embeddings)

# Group sentences by cluster
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(sentences[i])

for cluster_id, cluster_sentences in clusters.items():
    print(f"\nCluster {cluster_id}:")
    for sent in cluster_sentences:
        print(f"  - {sent}")

# 4. Paraphrase detection
print("\n=== Paraphrase Detection ===")
threshold = 0.8
print(f"Using similarity threshold: {threshold}")
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        is_paraphrase = sim > threshold
        print(f"Sentences {i+1} & {j+1}: Similarity {sim:.4f}, Paraphrase: {is_paraphrase}")

# 5. Semantic search example
print("\n=== Semantic Search Example ===")
query = "What is machine learning?"
try:
    query_embedding = sbert.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    most_similar_idx = similarities.argmax()
    print(f"Query: {query}")
    print(f"Most similar sentence: {sentences[most_similar_idx]}")
    print(f"Similarity score: {similarities[most_similar_idx]:.4f}")
except:
    print("Semantic search requires SBERT model to be loaded.")

# 6. Topic-based semantic analysis
print("\n=== Topic-based Semantic Analysis ===")
# Define topic keywords
topics = {
    "Machine Learning": ["machine learning", "artificial intelligence", "neural networks", "deep learning"],
    "Weather": ["weather", "forecast", "rain", "sunny"],
    "Data Science": ["data science", "statistics", "programming", "insights"]
}

# Find sentences related to each topic
for topic_name, keywords in topics.items():
    print(f"\nTopic: {topic_name}")
    related_sentences = []
    for i, sentence in enumerate(sentences):
        if any(keyword in sentence.lower() for keyword in keywords):
            related_sentences.append((i+1, sentence))
    
    for idx, sent in related_sentences:
        print(f"  {idx}. {sent}")

# 7. Semantic diversity analysis
print("\n=== Semantic Diversity Analysis ===")
# Calculate average similarity within clusters
for cluster_id in range(3):
    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
    if len(cluster_indices) > 1:
        cluster_embeddings = embeddings[cluster_indices]
        avg_similarity = np.mean(cosine_similarity(cluster_embeddings))
        print(f"Cluster {cluster_id} average similarity: {avg_similarity:.4f}")
        print(f"Cluster {cluster_id} size: {len(cluster_indices)} sentences") 