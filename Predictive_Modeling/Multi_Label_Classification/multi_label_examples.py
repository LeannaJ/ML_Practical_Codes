"""
Multi-Label Classification Examples
===================================

- Document Classification (News Articles)
- Image Tagging (Scene Recognition)
- Music Genre Classification
- Model comparison and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           hamming_loss, multilabel_confusion_matrix, classification_report)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.filterwarnings('ignore')

# For deep learning-based multi-label classification
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# 1. Document Classification (News Articles)
print("=== Document Classification (News Articles) ===")

def generate_news_data(n_articles=3000):
    """Generate synthetic news article data with multiple labels"""
    np.random.seed(42)
    
    # Define topics and their related keywords
    topics = {
        'Technology': ['artificial intelligence', 'machine learning', 'blockchain', 'cybersecurity', 'cloud computing'],
        'Politics': ['election', 'government', 'policy', 'congress', 'democracy'],
        'Business': ['stock market', 'economy', 'investment', 'startup', 'finance'],
        'Sports': ['football', 'basketball', 'baseball', 'olympics', 'championship'],
        'Health': ['medical', 'healthcare', 'disease', 'vaccine', 'wellness'],
        'Entertainment': ['movie', 'music', 'celebrity', 'award', 'film']
    }
    
    # Generate articles
    articles = []
    labels = []
    
    for i in range(n_articles):
        # Randomly select 1-3 topics for each article
        num_topics = np.random.randint(1, 4)
        selected_topics = np.random.choice(list(topics.keys()), num_topics, replace=False)
        
        # Generate article text based on selected topics
        article_text = ""
        for topic in selected_topics:
            keywords = topics[topic]
            # Add 2-4 sentences for each topic
            num_sentences = np.random.randint(2, 5)
            for _ in range(num_sentences):
                keyword = np.random.choice(keywords)
                sentence_templates = [
                    f"The {keyword} industry is experiencing significant growth.",
                    f"Experts predict major changes in {keyword} this year.",
                    f"New developments in {keyword} are reshaping the landscape.",
                    f"Companies are investing heavily in {keyword} solutions.",
                    f"The future of {keyword} looks promising for investors."
                ]
                sentence = np.random.choice(sentence_templates)
                article_text += sentence + " "
        
        # Add some random words for variety
        random_words = ['innovation', 'development', 'research', 'analysis', 'trend', 'future', 'market']
        for _ in range(np.random.randint(3, 8)):
            word = np.random.choice(random_words)
            article_text += f"{word} "
        
        articles.append(article_text.strip())
        labels.append(selected_topics)
    
    return pd.DataFrame({
        'article_id': range(1, n_articles + 1),
        'text': articles,
        'topics': labels
    })

# Generate news data
news_data = generate_news_data()
print(f"News data shape: {news_data.shape}")

# Prepare features and labels
X_news = news_data['text']
y_news = news_data['topics']

# Convert multi-label to binary format
mlb_news = MultiLabelBinarizer()
y_news_binary = mlb_news.fit_transform(y_news)
print(f"Number of unique labels: {len(mlb_news.classes_)}")
print(f"Label classes: {mlb_news.classes_}")

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
X_news_tfidf = tfidf.fit_transform(X_news)

# Split data
X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(
    X_news_tfidf, y_news_binary, test_size=0.2, random_state=42
)

# 2. Image Tagging (Scene Recognition)
print("\n=== Image Tagging (Scene Recognition) ===")

def generate_image_data(n_images=2500):
    """Generate synthetic image scene data with multiple tags"""
    np.random.seed(42)
    
    # Define scene types and their characteristics
    scenes = {
        'Nature': ['mountain', 'forest', 'ocean', 'desert', 'river'],
        'Urban': ['city', 'building', 'street', 'traffic', 'skyscraper'],
        'Indoor': ['room', 'office', 'kitchen', 'bedroom', 'living'],
        'People': ['crowd', 'family', 'children', 'adults', 'group'],
        'Transportation': ['car', 'bus', 'train', 'airplane', 'bicycle'],
        'Weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'stormy']
    }
    
    # Generate image features (simulating extracted features from CNN)
    images = []
    labels = []
    
    for i in range(n_images):
        # Randomly select 1-4 scene types for each image
        num_scenes = np.random.randint(1, 5)
        selected_scenes = np.random.choice(list(scenes.keys()), num_scenes, replace=False)
        
        # Generate feature vector (simulating CNN features)
        features = []
        for scene in selected_scenes:
            # Each scene contributes to specific feature dimensions
            scene_features = np.random.normal(0.7, 0.2, 50)  # High activation for selected scenes
            features.extend(scene_features)
        
        # Add features for non-selected scenes (low activation)
        remaining_scenes = set(scenes.keys()) - set(selected_scenes)
        for scene in remaining_scenes:
            scene_features = np.random.normal(0.1, 0.1, 50)  # Low activation
            features.extend(scene_features)
        
        # Add some noise
        features = np.array(features) + np.random.normal(0, 0.05, len(features))
        features = np.clip(features, 0, 1)  # Normalize to [0, 1]
        
        images.append(features)
        labels.append(selected_scenes)
    
    return pd.DataFrame({
        'image_id': range(1, n_images + 1),
        'features': images,
        'scenes': labels
    })

# Generate image data
image_data = generate_image_data()
print(f"Image data shape: {image_data.shape}")

# Prepare features and labels
X_image = np.array(image_data['features'].tolist())
y_image = image_data['scenes']

# Convert multi-label to binary format
mlb_image = MultiLabelBinarizer()
y_image_binary = mlb_image.fit_transform(y_image)
print(f"Number of unique labels: {len(mlb_image.classes_)}")
print(f"Label classes: {mlb_image.classes_}")

# Split data
X_train_image, X_test_image, y_train_image, y_test_image = train_test_split(
    X_image, y_image_binary, test_size=0.2, random_state=42
)

# Scale features
scaler_image = StandardScaler()
X_train_scaled_image = scaler_image.fit_transform(X_train_image)
X_test_scaled_image = scaler_image.transform(X_test_image)

# 3. Music Genre Classification
print("\n=== Music Genre Classification ===")

def generate_music_data(n_songs=2000):
    """Generate synthetic music data with multiple genres"""
    np.random.seed(42)
    
    # Define music genres and their characteristics
    genres = {
        'Rock': ['guitar', 'drums', 'electric', 'loud', 'energy'],
        'Pop': ['melody', 'catchy', 'vocals', 'radio', 'mainstream'],
        'Jazz': ['saxophone', 'improvisation', 'swing', 'complex', 'smooth'],
        'Classical': ['orchestra', 'piano', 'symphony', 'elegant', 'traditional'],
        'Electronic': ['synthesizer', 'beat', 'digital', 'dance', 'ambient'],
        'Hip-Hop': ['rap', 'beat', 'rhythm', 'urban', 'lyrics']
    }
    
    # Generate music features (simulating audio features)
    songs = []
    labels = []
    
    for i in range(n_songs):
        # Randomly select 1-3 genres for each song
        num_genres = np.random.randint(1, 4)
        selected_genres = np.random.choice(list(genres.keys()), num_genres, replace=False)
        
        # Generate feature vector (simulating audio features)
        features = []
        
        # Tempo (BPM)
        tempo = np.random.normal(120, 30)
        features.append(tempo)
        
        # Energy level
        energy = np.random.uniform(0, 1)
        features.append(energy)
        
        # Danceability
        danceability = np.random.uniform(0, 1)
        features.append(danceability)
        
        # Acousticness
        acousticness = np.random.uniform(0, 1)
        features.append(acousticness)
        
        # Instrumentalness
        instrumentalness = np.random.uniform(0, 1)
        features.append(instrumentalness)
        
        # Loudness
        loudness = np.random.uniform(-60, 0)
        features.append(loudness)
        
        # Valence (positivity)
        valence = np.random.uniform(0, 1)
        features.append(valence)
        
        # Duration
        duration = np.random.uniform(60, 300)
        features.append(duration)
        
        # Genre-specific features
        for genre in genres.keys():
            if genre in selected_genres:
                # High values for selected genres
                genre_features = np.random.normal(0.8, 0.1, 5)
            else:
                # Low values for non-selected genres
                genre_features = np.random.normal(0.2, 0.1, 5)
            features.extend(genre_features)
        
        songs.append(features)
        labels.append(selected_genres)
    
    return pd.DataFrame({
        'song_id': range(1, n_songs + 1),
        'features': songs,
        'genres': labels
    })

# Generate music data
music_data = generate_music_data()
print(f"Music data shape: {music_data.shape}")

# Prepare features and labels
X_music = np.array(music_data['features'].tolist())
y_music = music_data['genres']

# Convert multi-label to binary format
mlb_music = MultiLabelBinarizer()
y_music_binary = mlb_music.fit_transform(y_music)
print(f"Number of unique labels: {len(mlb_music.classes_)}")
print(f"Label classes: {mlb_music.classes_}")

# Split data
X_train_music, X_test_music, y_train_music, y_test_music = train_test_split(
    X_music, y_music_binary, test_size=0.2, random_state=42
)

# Scale features
scaler_music = StandardScaler()
X_train_scaled_music = scaler_music.fit_transform(X_train_music)
X_test_scaled_music = scaler_music.transform(X_test_music)

# 4. Multi-Label Classification Models
print("\n=== Multi-Label Classification Models ===")

def train_multi_label_models(X_train, y_train, X_test, y_test, dataset_name):
    """Train multiple multi-label classification models"""
    models = {
        'Random Forest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)),
        'Logistic Regression': MultiOutputClassifier(LogisticRegression(random_state=42)),
        'SVM': MultiOutputClassifier(SVC(probability=True, random_state=42)),
        'KNN': MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
        'Neural Network': MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42))
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} for {dataset_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        hamming = hamming_loss(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hamming_loss': hamming,
            'predictions': y_pred
        }
    
    return results

# Train models for each dataset
news_results = train_multi_label_models(X_train_news, y_train_news, X_test_news, y_test_news, "News Classification")
image_results = train_multi_label_models(X_train_scaled_image, y_train_image, X_test_scaled_image, y_test_image, "Image Tagging")
music_results = train_multi_label_models(X_train_scaled_music, y_train_music, X_test_scaled_music, y_test_music, "Music Classification")

# 5. Deep Learning Model (if TensorFlow available)
if TENSORFLOW_AVAILABLE:
    print("\n=== Deep Learning Multi-Label Classification ===")
    
    def create_deep_learning_model(input_dim, output_dim):
        """Create deep learning model for multi-label classification"""
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_dim, activation='sigmoid')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    # Train deep learning models
    dl_results = {}
    
    # News classification
    print("Training deep learning model for news classification...")
    dl_news_model = create_deep_learning_model(X_train_news.shape[1], y_train_news.shape[1])
    dl_news_model.fit(X_train_news.toarray(), y_train_news, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    y_pred_dl_news = (dl_news_model.predict(X_test_news.toarray()) > 0.5).astype(int)
    
    dl_results['News_DL'] = {
        'accuracy': accuracy_score(y_test_news, y_pred_dl_news),
        'precision': precision_score(y_test_news, y_pred_dl_news, average='micro'),
        'recall': recall_score(y_test_news, y_pred_dl_news, average='micro'),
        'f1': f1_score(y_test_news, y_pred_dl_news, average='micro'),
        'hamming_loss': hamming_loss(y_test_news, y_pred_dl_news),
        'predictions': y_pred_dl_news
    }
    
    # Image tagging
    print("Training deep learning model for image tagging...")
    dl_image_model = create_deep_learning_model(X_train_scaled_image.shape[1], y_train_image.shape[1])
    dl_image_model.fit(X_train_scaled_image, y_train_image, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    y_pred_dl_image = (dl_image_model.predict(X_test_scaled_image) > 0.5).astype(int)
    
    dl_results['Image_DL'] = {
        'accuracy': accuracy_score(y_test_image, y_pred_dl_image),
        'precision': precision_score(y_test_image, y_pred_dl_image, average='micro'),
        'recall': recall_score(y_test_image, y_pred_dl_image, average='micro'),
        'f1': f1_score(y_test_image, y_pred_dl_image, average='micro'),
        'hamming_loss': hamming_loss(y_test_image, y_pred_dl_image),
        'predictions': y_pred_dl_image
    }
    
    # Music classification
    print("Training deep learning model for music classification...")
    dl_music_model = create_deep_learning_model(X_train_scaled_music.shape[1], y_train_music.shape[1])
    dl_music_model.fit(X_train_scaled_music, y_train_music, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
    y_pred_dl_music = (dl_music_model.predict(X_test_scaled_music) > 0.5).astype(int)
    
    dl_results['Music_DL'] = {
        'accuracy': accuracy_score(y_test_music, y_pred_dl_music),
        'precision': precision_score(y_test_music, y_pred_dl_music, average='micro'),
        'recall': recall_score(y_test_music, y_pred_dl_music, average='micro'),
        'f1': f1_score(y_test_music, y_pred_dl_music, average='micro'),
        'hamming_loss': hamming_loss(y_test_music, y_pred_dl_music),
        'predictions': y_pred_dl_music
    }

# 6. Visualization and Comparison
print("\n=== Visualization and Comparison ===")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Function to plot results
def plot_multi_label_results(results, title, ax, metric='f1'):
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    bars = ax.bar(models, values, alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{title} - {metric.upper()}')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')

# Plot F1 scores
plot_multi_label_results(news_results, 'News Classification', axes[0, 0], 'f1')
plot_multi_label_results(image_results, 'Image Tagging', axes[0, 1], 'f1')
plot_multi_label_results(music_results, 'Music Classification', axes[0, 2], 'f1')

# Plot Hamming Loss (lower is better)
def plot_hamming_loss(results, title, ax):
    models = list(results.keys())
    values = [results[model]['hamming_loss'] for model in models]
    
    bars = ax.bar(models, values, alpha=0.8, color='orange')
    ax.set_xlabel('Models')
    ax.set_ylabel('Hamming Loss')
    ax.set_title(f'{title} - Hamming Loss')
    ax.set_xticklabels(models, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{value:.3f}', ha='center', va='bottom')

plot_hamming_loss(news_results, 'News Classification', axes[1, 0])
plot_hamming_loss(image_results, 'Image Tagging', axes[1, 1])
plot_hamming_loss(music_results, 'Music Classification', axes[1, 2])

plt.tight_layout()
plt.show()

# 7. Label Distribution Analysis
print("\n=== Label Distribution Analysis ===")

# Analyze label distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# News topics distribution
news_label_counts = np.sum(y_news_binary, axis=0)
axes[0].bar(mlb_news.classes_, news_label_counts, alpha=0.7)
axes[0].set_xlabel('Topics')
axes[0].set_ylabel('Count')
axes[0].set_title('News Topics Distribution')
axes[0].tick_params(axis='x', rotation=45)

# Image scenes distribution
image_label_counts = np.sum(y_image_binary, axis=0)
axes[1].bar(mlb_image.classes_, image_label_counts, alpha=0.7)
axes[1].set_xlabel('Scenes')
axes[1].set_ylabel('Count')
axes[1].set_title('Image Scenes Distribution')
axes[1].tick_params(axis='x', rotation=45)

# Music genres distribution
music_label_counts = np.sum(y_music_binary, axis=0)
axes[2].bar(mlb_music.classes_, music_label_counts, alpha=0.7)
axes[2].set_xlabel('Genres')
axes[2].set_ylabel('Count')
axes[2].set_title('Music Genres Distribution')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 8. Multi-Label Correlation Analysis
print("\n=== Multi-Label Correlation Analysis ===")

# Create correlation matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# News topics correlation
news_corr = np.corrcoef(y_news_binary.T)
im1 = axes[0].imshow(news_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(mlb_news.classes_)))
axes[0].set_yticks(range(len(mlb_news.classes_)))
axes[0].set_xticklabels(mlb_news.classes_, rotation=45)
axes[0].set_yticklabels(mlb_news.classes_)
axes[0].set_title('News Topics Correlation')
plt.colorbar(im1, ax=axes[0])

# Image scenes correlation
image_corr = np.corrcoef(y_image_binary.T)
im2 = axes[1].imshow(image_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_xticks(range(len(mlb_image.classes_)))
axes[1].set_yticks(range(len(mlb_image.classes_)))
axes[1].set_xticklabels(mlb_image.classes_, rotation=45)
axes[1].set_yticklabels(mlb_image.classes_)
axes[1].set_title('Image Scenes Correlation')
plt.colorbar(im2, ax=axes[1])

# Music genres correlation
music_corr = np.corrcoef(y_music_binary.T)
im3 = axes[2].imshow(music_corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[2].set_xticks(range(len(mlb_music.classes_)))
axes[2].set_yticks(range(len(mlb_music.classes_)))
axes[2].set_xticklabels(mlb_music.classes_, rotation=45)
axes[2].set_yticklabels(mlb_music.classes_)
axes[2].set_title('Music Genres Correlation')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()

# 9. Sample Predictions
print("\n=== Sample Predictions ===")

# Function to show sample predictions
def show_sample_predictions(X_test, y_test, y_pred, mlb, dataset_name, n_samples=5):
    """Show sample predictions for multi-label classification"""
    print(f"\n{dataset_name} - Sample Predictions:")
    print("-" * 50)
    
    for i in range(min(n_samples, len(X_test))):
        true_labels = mlb.inverse_transform([y_test[i]])[0]
        pred_labels = mlb.inverse_transform([y_pred[i]])[0]
        
        print(f"Sample {i+1}:")
        print(f"  True labels: {true_labels}")
        print(f"  Predicted labels: {pred_labels}")
        print(f"  Correct: {set(true_labels) == set(pred_labels)}")
        print()

# Show sample predictions for each dataset
show_sample_predictions(X_test_news, y_test_news, news_results['Random Forest']['predictions'], 
                       mlb_news, "News Classification")
show_sample_predictions(X_test_scaled_image, y_test_image, image_results['Random Forest']['predictions'], 
                       mlb_image, "Image Tagging")
show_sample_predictions(X_test_scaled_music, y_test_music, music_results['Random Forest']['predictions'], 
                       mlb_music, "Music Classification")

# 10. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

# Find best model for each problem
def find_best_multi_label_model(results, metric='f1'):
    best_model = max(results.items(), key=lambda x: x[1][metric])
    return best_model[0], best_model[1]

print("Best Models by F1 Score:")
print(f"1. News Classification:")
best_news, news_metrics = find_best_multi_label_model(news_results)
print(f"   Best model: {best_news}")
print(f"   F1: {news_metrics['f1']:.3f}")
print(f"   Hamming Loss: {news_metrics['hamming_loss']:.3f}")

print(f"\n2. Image Tagging:")
best_image, image_metrics = find_best_multi_label_model(image_results)
print(f"   Best model: {best_image}")
print(f"   F1: {image_metrics['f1']:.3f}")
print(f"   Hamming Loss: {image_metrics['hamming_loss']:.3f}")

print(f"\n3. Music Classification:")
best_music, music_metrics = find_best_multi_label_model(music_results)
print(f"   Best model: {best_music}")
print(f"   F1: {music_metrics['f1']:.3f}")
print(f"   Hamming Loss: {music_metrics['hamming_loss']:.3f}")

if TENSORFLOW_AVAILABLE:
    print(f"\n4. Deep Learning Results:")
    for name, metrics in dl_results.items():
        print(f"   {name}: F1={metrics['f1']:.3f}, Hamming Loss={metrics['hamming_loss']:.3f}")

print(f"\nKey Insights:")
print(f"- Random Forest performs well across all multi-label problems")
print(f"- Hamming Loss is a better metric for multi-label evaluation")
print(f"- Label correlations affect model performance")
print(f"- Feature engineering is crucial for multi-label classification")

print(f"\nRecommendations:")
print(f"- Use Random Forest for balanced performance and interpretability")
print(f"- Use Neural Networks for complex, non-linear relationships")
print(f"- Consider label correlations in feature engineering")
print(f"- Use appropriate evaluation metrics (Hamming Loss, F1-micro)")
print(f"- Handle class imbalance in multi-label scenarios")
print(f"- Consider label-specific thresholds for better performance") 