"""
Text Preprocessing Examples
==========================

- Basic text cleaning
- Tokenization (word, sentence)
- Stopword removal
- Lemmatization & stemming
- POS tagging
- Named Entity Recognition
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

# 1. Basic text cleaning
text = """Hello! This is a sample text for natural language processing. 
Let's clean, tokenize, and analyze it. The quick brown fox jumps over the lazy dog. 
Machine learning is fascinating and has many applications in NLP."""
print("Original text:", text)

# Lowercase
text_clean = text.lower()
# Remove punctuation
text_clean = re.sub(f"[{re.escape(string.punctuation)}]", "", text_clean)
# Remove numbers
text_clean = re.sub(r"\d+", "", text_clean)
# Remove extra whitespace
text_clean = re.sub(r"\s+", " ", text_clean).strip()
print("\nCleaned text:", text_clean)

# 2. Tokenization
words = word_tokenize(text_clean)
sentences = sent_tokenize(text)
print("\nWord tokens:", words)
print("Sentence tokens:", sentences)

# 3. Stopword removal
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w not in stop_words]
print("\nWords after stopword removal:", filtered_words)

# 4. Stemming & Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed = [stemmer.stem(w) for w in filtered_words]
lemmatized = [lemmatizer.lemmatize(w) for w in filtered_words]
print("\nStemmed words:", stemmed)
print("Lemmatized words:", lemmatized)

# 5. POS tagging (English)
pos_tags = nltk.pos_tag(filtered_words)
print("\nPOS tags:", pos_tags)

# 6. Named Entity Recognition
ner_text = "John Smith works at Google in New York City. He loves machine learning."
ner_tokens = word_tokenize(ner_text)
ner_pos = nltk.pos_tag(ner_tokens)
ner_chunks = ne_chunk(ner_pos)
print("\nNamed Entity Recognition:")
for chunk in ner_chunks:
    if hasattr(chunk, 'label'):
        print(f"{chunk.label()}: {' '.join(c[0] for c in chunk)}")

# 7. Word frequency
freq = Counter(filtered_words)
print("\nWord frequency:", freq)

# 8. Advanced text cleaning function
def clean_text_advanced(text):
    """Advanced text cleaning with multiple steps"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep apostrophes for contractions
    text = re.sub(r'[^\w\s\']', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Test advanced cleaning
sample_text = "Check out this link: https://example.com and email me at john@email.com! #NLP #MachineLearning"
cleaned_advanced = clean_text_advanced(sample_text)
print(f"\nAdvanced cleaning:")
print(f"Original: {sample_text}")
print(f"Cleaned: {cleaned_advanced}") 