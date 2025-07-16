"""
Topic Modeling Examples
======================

- LDA (Latent Dirichlet Allocation)
- NMF (Non-negative Matrix Factorization)
- Topic visualization (pyLDAvis)
- Real-world news article data usage
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.sklearn

# 1. Data preparation (20 Newsgroups)
print("\n=== Load 20 Newsgroups Data ===")
news = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
docs = news.data[:1000]  # Use subset for speed

# 2. TF-IDF & Count Vectorization
print("\n=== Vectorization ===")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(docs)
count = CountVectorizer(max_features=2000, stop_words='english')
count_matrix = count.fit_transform(docs)

# 3. LDA
print("\n=== LDA Topic Modeling ===")
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda_topics = lda.fit_transform(count_matrix)

print("Top words per LDA topic:")
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx+1}: ", end="")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
print_top_words(lda, count.get_feature_names_out())

# 4. NMF
print("\n=== NMF Topic Modeling ===")
nmf = NMF(n_components=5, random_state=42)
nmf_topics = nmf.fit_transform(tfidf_matrix)

print("Top words per NMF topic:")
print_top_words(nmf, tfidf.get_feature_names_out())

# 5. pyLDAvis visualization (LDA)
print("\n=== pyLDAvis Visualization (LDA) ===")
try:
    vis = pyLDAvis.sklearn.prepare(lda, count_matrix, count, mds='tsne')
    pyLDAvis.show(vis)
except Exception as e:
    print("pyLDAvis visualization only works in Jupyter environment or requires a browser.")
    print("Error:", e)

# 6. Custom English corpus for topic modeling
print("\n=== Custom English Corpus Topic Modeling ===")

# Sample English documents for topic modeling
custom_docs = [
    "Machine learning algorithms are transforming the way we process data and make predictions.",
    "Deep learning models like neural networks have revolutionized artificial intelligence applications.",
    "Natural language processing enables computers to understand and generate human language.",
    "Computer vision systems can identify objects and patterns in images and videos.",
    "Data science combines statistics, programming, and domain expertise to extract insights.",
    "Big data technologies handle massive datasets for analysis and decision making.",
    "Cloud computing provides scalable infrastructure for deploying machine learning models.",
    "Internet of Things devices generate vast amounts of sensor data for analysis.",
    "Cybersecurity measures protect systems from digital threats and attacks.",
    "Blockchain technology creates secure and transparent digital transaction records.",
    "Mobile app development focuses on creating user-friendly applications for smartphones.",
    "Web development involves building interactive websites and web applications.",
    "Database management systems organize and retrieve information efficiently.",
    "Software engineering principles guide the development of reliable software systems.",
    "Human-computer interaction studies how people interact with technology interfaces.",
    "Robotics combines mechanical engineering with artificial intelligence for automation.",
    "Virtual reality creates immersive digital environments for entertainment and training.",
    "Augmented reality overlays digital information onto the real world.",
    "Quantum computing explores new computational paradigms using quantum mechanics.",
    "Edge computing processes data closer to the source for faster response times."
]

# Vectorize custom documents
custom_tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
custom_tfidf_matrix = custom_tfidf.fit_transform(custom_docs)

custom_count = CountVectorizer(max_features=1000, stop_words='english')
custom_count_matrix = custom_count.fit_transform(custom_docs)

# LDA on custom data
custom_lda = LatentDirichletAllocation(n_components=4, random_state=42)
custom_lda_topics = custom_lda.fit_transform(custom_count_matrix)

print("Top words per LDA topic (Custom Data):")
print_top_words(custom_lda, custom_count.get_feature_names_out())

# NMF on custom data
custom_nmf = NMF(n_components=4, random_state=42)
custom_nmf_topics = custom_nmf.fit_transform(custom_tfidf_matrix)

print("\nTop words per NMF topic (Custom Data):")
print_top_words(custom_nmf, custom_tfidf.get_feature_names_out())

# 7. Topic distribution analysis
print("\n=== Topic Distribution Analysis ===")
doc_topics = custom_lda.transform(custom_count_matrix)
print("Document-topic distribution (first 5 documents):")
for i in range(5):
    print(f"Document {i+1}: {doc_topics[i]}")

# Find documents most similar to each topic
print("\nMost representative documents for each topic:")
for topic_idx in range(4):
    topic_docs = doc_topics[:, topic_idx]
    top_doc_idx = topic_docs.argmax()
    print(f"Topic {topic_idx+1}: Document {top_doc_idx+1}")
    print(f"Text: {custom_docs[top_doc_idx][:100]}...")
    print() 