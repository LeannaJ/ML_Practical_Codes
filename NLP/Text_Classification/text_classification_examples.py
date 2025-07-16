"""
Text Classification Examples
===========================

- Sentiment analysis (IMDB dataset)
- News classification (20 Newsgroups)
- Pipeline: TF-IDF + LogisticRegression
- Pipeline: Word2Vec + RandomForest
- Simple neural network text classifier
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import gensim
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Sentiment analysis (IMDB dataset, keras)
print("\n=== Sentiment Analysis (IMDB, Keras) ===")
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

x_train_text = [decode_review(x) for x in x_train[:2000]]
y_train_small = y_train[:2000]
x_test_text = [decode_review(x) for x in x_test[:500]]
y_test_small = y_test[:500]

# TF-IDF + LogisticRegression
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=200))
])
pipe.fit(x_train_text, y_train_small)
y_pred = pipe.predict(x_test_text)
print("TF-IDF + LogisticRegression accuracy:", accuracy_score(y_test_small, y_pred))
print(classification_report(y_test_small, y_pred))

# 2. News classification (20 Newsgroups)
print("\n=== News Classification (20 Newsgroups) ===")
news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.2, random_state=42)

pipe_news = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=200))
])
pipe_news.fit(X_train[:2000], y_train[:2000])
y_pred_news = pipe_news.predict(X_test[:500])
print("TF-IDF + LogisticRegression accuracy:", accuracy_score(y_test[:500], y_pred_news))
print(classification_report(y_test[:500], y_pred_news, target_names=news.target_names))

# 3. Word2Vec + RandomForest (on IMDB small)
print("\n=== Word2Vec + RandomForest (IMDB small) ===")
tokenized = [t.split() for t in x_train_text]
w2v = Word2Vec(sentences=tokenized, vector_size=50, window=3, min_count=1, workers=1, seed=42)
def get_w2v_features(texts, model):
    features = []
    for sent in texts:
        words = sent.split()
        vecs = [model.wv[w] for w in words if w in model.wv]
        if vecs:
            features.append(np.mean(vecs, axis=0))
        else:
            features.append(np.zeros(model.vector_size))
    return np.array(features)
X_train_w2v = get_w2v_features(x_train_text, w2v)
X_test_w2v = get_w2v_features(x_test_text, w2v)
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_w2v, y_train_small)
y_pred_rf = rf.predict(X_test_w2v)
print("Word2Vec + RandomForest accuracy:", accuracy_score(y_test_small, y_pred_rf))

# 4. Simple neural network text classifier (Keras)
print("\n=== Simple Neural Network Text Classifier ===")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train_text)
X_train_seq = tokenizer.texts_to_sequences(x_train_text)
X_test_seq = tokenizer.texts_to_sequences(x_test_text)
X_train_pad = pad_sequences(X_train_seq, maxlen=200)
X_test_pad = pad_sequences(X_test_seq, maxlen=200)

model = Sequential([
    Embedding(5000, 32, input_length=200),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_pad, y_train_small, epochs=3, batch_size=32, validation_split=0.2, verbose=2)
loss, acc = model.evaluate(X_test_pad, y_test_small, verbose=0)
print(f"Neural Net Test Accuracy: {acc:.4f}")

# 5. Custom text classification with English sample data
print("\n=== Custom Text Classification (English Sample Data) ===")

# Sample English text data for classification
sample_texts = [
    "This movie is absolutely fantastic and I loved every minute of it!",
    "The acting was terrible and the plot made no sense at all.",
    "Great performance by the lead actor, highly recommended!",
    "Boring and predictable, waste of time and money.",
    "Amazing cinematography and brilliant storytelling.",
    "The worst film I have ever seen, complete disaster.",
    "Outstanding direction and compelling narrative.",
    "Poor script and weak character development.",
    "Incredible special effects and engaging storyline.",
    "Disappointing ending and lackluster performances.",
    "Masterpiece of modern cinema, truly inspiring.",
    "Confusing plot and terrible dialogue throughout.",
    "Beautiful soundtrack and emotional depth.",
    "Generic and forgettable, nothing special.",
    "Revolutionary filmmaking techniques and bold vision."
]

# Labels: 1 for positive, 0 for negative
sample_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# Split the data
X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
    sample_texts, sample_labels, test_size=0.3, random_state=42, stratify=sample_labels
)

# TF-IDF + LogisticRegression on custom data
custom_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=200))
])
custom_pipe.fit(X_train_custom, y_train_custom)
y_pred_custom = custom_pipe.predict(X_test_custom)
print("Custom data - TF-IDF + LogisticRegression accuracy:", accuracy_score(y_test_custom, y_pred_custom))
print(classification_report(y_test_custom, y_pred_custom, target_names=['Negative', 'Positive']))

# Test on new examples
test_examples = [
    "This is the best movie ever made!",
    "I hated this film, it was awful.",
    "The story was engaging and well-written."
]

predictions = custom_pipe.predict(test_examples)
probabilities = custom_pipe.predict_proba(test_examples)

print("\nPredictions on new examples:")
for text, pred, prob in zip(test_examples, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[1] if pred == 1 else prob[0]
    print(f"Text: '{text}'")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.3f})")
    print() 