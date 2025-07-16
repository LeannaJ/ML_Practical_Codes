"""
Word Embeddings Examples
=======================

- TF-IDF Vectorization
- Word2Vec (gensim)
- FastText (gensim)
- GloVe (pretrained)
- Embedding visualization (t-SNE)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec, FastText
import os
import urllib.request
import zipfile

# 1. Sample data - English corpus
corpus = [
    "Natural language processing is a subfield of artificial intelligence.",
    "Word embeddings represent words as dense vectors in continuous space.",
    "Machine learning algorithms can learn patterns from text data.",
    "Deep learning models like BERT and GPT have revolutionized NLP.",
    "Text classification and sentiment analysis are common NLP tasks.",
    "Information retrieval systems use vector similarity to find relevant documents.",
    "Named entity recognition identifies people, places, and organizations in text.",
    "Language models can generate human-like text and understand context."
]

# 2. TF-IDF Vectorization
print("\n=== TF-IDF Vectorization ===")
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
print("TF-IDF feature names:", tfidf.get_feature_names_out())
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# 3. Word2Vec (gensim)
print("\n=== Word2Vec (gensim) ===")
tokenized = [s.lower().split() for s in corpus]
w2v_model = Word2Vec(sentences=tokenized, vector_size=50, window=3, min_count=1, workers=1, seed=42)
print("Word2Vec vector for 'embeddings':", w2v_model.wv['embeddings'][:5])

# 4. FastText (gensim)
print("\n=== FastText (gensim) ===")
ft_model = FastText(sentences=tokenized, vector_size=50, window=3, min_count=1, workers=1, seed=42)
print("FastText vector for 'embeddings':", ft_model.wv['embeddings'][:5])

# 5. GloVe (pretrained, 50d)
def download_glove():
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    fname = "glove.6B.zip"
    if not os.path.exists(fname):
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(url, fname)
    if not os.path.exists("glove.6B.50d.txt"):
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall()

def load_glove_embeddings(glove_file="glove.6B.50d.txt"):
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

print("\n=== GloVe (pretrained) ===")
try:
    download_glove()
    glove = load_glove_embeddings()
    print("GloVe vector for 'embeddings':", glove.get('embeddings', None)[:5])
except Exception as e:
    print("GloVe download/loading failed:", e)

# 6. Embedding visualization (t-SNE)
print("\n=== Embedding Visualization (t-SNE) ===")
words = ['embeddings', 'nlp', 'word2vec', 'fasttext', 'glove', 'machine', 'learning', 'text', 'language', 'processing']
vectors = []
labels = []
for w in words:
    if w in w2v_model.wv:
        vectors.append(w2v_model.wv[w])
        labels.append(w + ' (w2v)')
    if w in ft_model.wv:
        vectors.append(ft_model.wv[w])
        labels.append(w + ' (ft)')
    if 'glove' in locals() and w in glove:
        vectors.append(glove[w])
        labels.append(w + ' (glove)')

if len(vectors) > 0:
    vectors = np.array(vectors)
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    reduced = tsne.fit_transform(vectors)
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0]+0.01, reduced[i, 1]+0.01, label, fontsize=9)
    plt.title('Word Embeddings Visualization (t-SNE)')
    plt.grid(True)
    plt.show()
else:
    print("No vectors available for visualization.")

# 7. Word similarity examples
print("\n=== Word Similarity Examples ===")
if 'glove' in locals():
    def find_similar_words(word, top_n=5):
        if word in glove:
            word_vec = glove[word]
            similarities = {}
            for other_word, other_vec in glove.items():
                if other_word != word:
                    similarity = np.dot(word_vec, other_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
                    similarities[other_word] = similarity
            return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return []
    
    print("Words similar to 'computer':")
    similar_words = find_similar_words('computer')
    for word, sim in similar_words:
        print(f"  {word}: {sim:.3f}")
    
    print("\nWords similar to 'language':")
    similar_words = find_similar_words('language')
    for word, sim in similar_words:
        print(f"  {word}: {sim:.3f}") 