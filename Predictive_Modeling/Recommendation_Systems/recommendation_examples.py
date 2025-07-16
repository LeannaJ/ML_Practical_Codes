"""
Recommendation Systems Examples
===============================

- Movie Recommendations (Collaborative Filtering, Content-Based)
- Product Recommendations (Matrix Factorization, Neural Networks)
- Music Recommendations (Hybrid Methods)
- Evaluation and comparison of different approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# For neural network-based recommendations
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Install with: pip install tensorflow")

# 1. Movie Recommendations
print("=== Movie Recommendations ===")

def generate_movie_data(n_users=1000, n_movies=500, sparsity=0.95):
    """Generate synthetic movie rating data"""
    np.random.seed(42)
    
    # Generate movie features
    movie_genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Romance', 'Thriller', 'Documentary']
    movie_years = np.random.randint(1980, 2023, n_movies)
    movie_ratings = np.random.uniform(1, 10, n_movies)
    
    # Generate movies with genre combinations
    movies = []
    for i in range(n_movies):
        # Each movie can have 1-3 genres
        num_genres = np.random.randint(1, 4)
        genres = np.random.choice(movie_genres, num_genres, replace=False)
        movies.append({
            'movie_id': i + 1,
            'title': f'Movie_{i+1}',
            'year': movie_years[i],
            'rating': movie_ratings[i],
            'genres': '|'.join(genres)
        })
    
    movies_df = pd.DataFrame(movies)
    
    # Generate user-movie ratings
    ratings = []
    n_ratings = int(n_users * n_movies * (1 - sparsity))
    
    for _ in range(n_ratings):
        user_id = np.random.randint(1, n_users + 1)
        movie_id = np.random.randint(1, n_movies + 1)
        
        # Generate rating based on movie quality and user preferences
        movie_rating = movies_df.loc[movie_id-1, 'rating']
        user_bias = np.random.normal(0, 1)  # User-specific bias
        rating = movie_rating + user_bias + np.random.normal(0, 1)
        rating = np.clip(rating, 1, 10)
        
        ratings.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': np.random.randint(1000000000, 1600000000)
        })
    
    ratings_df = pd.DataFrame(ratings)
    
    # Generate user features
    users = []
    for i in range(n_users):
        age = np.random.randint(16, 80)
        gender = np.random.choice(['M', 'F'])
        occupation = np.random.choice(['student', 'engineer', 'teacher', 'doctor', 'other'])
        
        users.append({
            'user_id': i + 1,
            'age': age,
            'gender': gender,
            'occupation': occupation
        })
    
    users_df = pd.DataFrame(users)
    
    return movies_df, users_df, ratings_df

# Generate movie data
movies_df, users_df, ratings_df = generate_movie_data()
print(f"Movies: {len(movies_df)}")
print(f"Users: {len(users_df)}")
print(f"Ratings: {len(ratings_df)}")
print(f"Sparsity: {1 - len(ratings_df) / (len(users_df) * len(movies_df)):.3f}")

# Collaborative Filtering - User-Based
print("\n--- User-Based Collaborative Filtering ---")

def user_based_cf(ratings_matrix, user_id, n_recommendations=5):
    """User-based collaborative filtering"""
    if user_id not in ratings_matrix.index:
        return []
    
    # Find similar users
    user_ratings = ratings_matrix.loc[user_id]
    similarities = cosine_similarity([user_ratings], ratings_matrix)[0]
    
    # Get top similar users (excluding self)
    similar_users = np.argsort(similarities)[::-1][1:11]  # Top 10 similar users
    
    # Get movies rated by similar users but not by target user
    user_rated = set(ratings_matrix.columns[user_ratings.notna()])
    recommendations = {}
    
    for similar_user in similar_users:
        similar_user_id = ratings_matrix.index[similar_user]
        similar_user_ratings = ratings_matrix.loc[similar_user_id]
        
        for movie_id in ratings_matrix.columns:
            if movie_id not in user_rated and pd.notna(similar_user_ratings[movie_id]):
                if movie_id not in recommendations:
                    recommendations[movie_id] = []
                recommendations[movie_id].append(similar_user_ratings[movie_id])
    
    # Calculate average ratings and sort
    avg_ratings = {movie: np.mean(ratings) for movie, ratings in recommendations.items()}
    sorted_recommendations = sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_recommendations[:n_recommendations]

# Create ratings matrix
ratings_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')
print(f"Ratings matrix shape: {ratings_matrix.shape}")

# Test user-based CF
test_user = ratings_matrix.index[0]
recommendations = user_based_cf(ratings_matrix, test_user)
print(f"\nRecommendations for user {test_user}:")
for movie_id, avg_rating in recommendations:
    movie_title = movies_df.loc[movie_id-1, 'title']
    print(f"  {movie_title}: {avg_rating:.2f}")

# Content-Based Filtering
print("\n--- Content-Based Filtering ---")

def content_based_filtering(movies_df, user_ratings, n_recommendations=5):
    """Content-based filtering using movie features"""
    # Create TF-IDF features from genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'].fillna(''))
    
    # Calculate movie similarities
    movie_similarities = cosine_similarity(tfidf_matrix)
    
    # Get user's rated movies
    rated_movies = user_ratings[user_ratings.notna()].index.tolist()
    
    if not rated_movies:
        return []
    
    # Calculate user profile (average of rated movies)
    user_profile = np.mean([movie_similarities[movie_id-1] for movie_id in rated_movies], axis=0)
    
    # Find movies not rated by user
    unrated_movies = [i for i in range(len(movies_df)) if i+1 not in rated_movies]
    
    # Calculate similarity scores
    movie_scores = [(movie_id+1, user_profile[movie_id]) for movie_id in unrated_movies]
    movie_scores.sort(key=lambda x: x[1], reverse=True)
    
    return movie_scores[:n_recommendations]

# Test content-based filtering
test_user_ratings = ratings_matrix.loc[test_user]
content_recommendations = content_based_filtering(movies_df, test_user_ratings)
print(f"\nContent-based recommendations for user {test_user}:")
for movie_id, score in content_recommendations:
    movie_title = movies_df.loc[movie_id-1, 'title']
    print(f"  {movie_title}: {score:.3f}")

# Matrix Factorization (SVD)
print("\n--- Matrix Factorization (SVD) ---")

def matrix_factorization_svd(ratings_matrix, n_components=50):
    """Matrix factorization using SVD"""
    # Fill NaN with 0 for SVD
    ratings_filled = ratings_matrix.fillna(0)
    
    # Apply SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(ratings_filled)
    movie_factors = svd.components_.T
    
    return user_factors, movie_factors, svd

# Apply SVD
user_factors, movie_factors, svd = matrix_factorization_svd(ratings_matrix)
print(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

def predict_rating_svd(user_id, movie_id, user_factors, movie_factors, ratings_matrix):
    """Predict rating using SVD factors"""
    if user_id not in ratings_matrix.index or movie_id not in ratings_matrix.columns:
        return 0
    
    user_idx = ratings_matrix.index.get_loc(user_id)
    movie_idx = ratings_matrix.columns.get_loc(movie_id)
    
    prediction = np.dot(user_factors[user_idx], movie_factors[movie_idx])
    return np.clip(prediction, 1, 10)

# Test SVD predictions
test_movie = ratings_matrix.columns[0]
predicted_rating = predict_rating_svd(test_user, test_movie, user_factors, movie_factors, ratings_matrix)
actual_rating = ratings_matrix.loc[test_user, test_movie]
print(f"\nSVD Prediction for user {test_user}, movie {test_movie}:")
print(f"  Predicted: {predicted_rating:.2f}")
print(f"  Actual: {actual_rating:.2f}" if pd.notna(actual_rating) else "  Actual: Not rated")

# 2. Product Recommendations
print("\n=== Product Recommendations ===")

def generate_product_data(n_users=800, n_products=400, n_purchases=5000):
    """Generate synthetic product purchase data"""
    np.random.seed(42)
    
    # Generate product features
    product_categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Food', 'Toys']
    product_prices = np.random.exponential(scale=50, size=n_products)
    product_prices = np.clip(product_prices, 5, 500)
    
    products = []
    for i in range(n_products):
        category = np.random.choice(product_categories)
        products.append({
            'product_id': i + 1,
            'name': f'Product_{i+1}',
            'category': category,
            'price': product_prices[i],
            'rating': np.random.uniform(3, 5)
        })
    
    products_df = pd.DataFrame(products)
    
    # Generate purchase history
    purchases = []
    for _ in range(n_purchases):
        user_id = np.random.randint(1, n_users + 1)
        product_id = np.random.randint(1, n_products + 1)
        quantity = np.random.randint(1, 5)
        
        purchases.append({
            'user_id': user_id,
            'product_id': product_id,
            'quantity': quantity,
            'timestamp': np.random.randint(1000000000, 1600000000)
        })
    
    purchases_df = pd.DataFrame(purchases)
    
    return products_df, purchases_df

# Generate product data
products_df, purchases_df = generate_product_data()
print(f"Products: {len(products_df)}")
print(f"Purchases: {len(purchases_df)}")

# Create purchase matrix (quantity as interaction strength)
purchase_matrix = purchases_df.groupby(['user_id', 'product_id'])['quantity'].sum().unstack(fill_value=0)
print(f"Purchase matrix shape: {purchase_matrix.shape}")

# Non-negative Matrix Factorization (NMF)
print("\n--- Non-negative Matrix Factorization ---")

def nmf_recommendations(purchase_matrix, n_components=20):
    """NMF for product recommendations"""
    nmf = NMF(n_components=n_components, random_state=42)
    user_factors = nmf.fit_transform(purchase_matrix)
    product_factors = nmf.components_
    
    return user_factors, product_factors, nmf

# Apply NMF
user_factors_nmf, product_factors_nmf, nmf_model = nmf_recommendations(purchase_matrix)
print(f"NMF reconstruction error: {nmf_model.reconstruction_err_:.2f}")

def get_nmf_recommendations(user_id, user_factors, product_factors, n_recommendations=5):
    """Get NMF-based recommendations"""
    if user_id not in purchase_matrix.index:
        return []
    
    user_idx = purchase_matrix.index.get_loc(user_id)
    user_profile = user_factors[user_idx]
    
    # Calculate predicted purchase quantities
    predictions = np.dot(user_profile, product_factors)
    
    # Get products not purchased by user
    user_purchases = purchase_matrix.loc[user_id]
    unrated_products = user_purchases[user_purchases == 0].index.tolist()
    
    # Get recommendations
    product_scores = [(product_id, predictions[product_id-1]) for product_id in unrated_products]
    product_scores.sort(key=lambda x: x[1], reverse=True)
    
    return product_scores[:n_recommendations]

# Test NMF recommendations
test_user_product = purchase_matrix.index[0]
nmf_recommendations = get_nmf_recommendations(test_user_product, user_factors_nmf, product_factors_nmf)
print(f"\nNMF recommendations for user {test_user_product}:")
for product_id, score in nmf_recommendations:
    product_name = products_df.loc[product_id-1, 'name']
    product_category = products_df.loc[product_id-1, 'category']
    print(f"  {product_name} ({product_category}): {score:.2f}")

# 3. Music Recommendations
print("\n=== Music Recommendations ===")

def generate_music_data(n_users=600, n_songs=300, n_listens=4000):
    """Generate synthetic music listening data"""
    np.random.seed(42)
    
    # Generate song features
    song_genres = ['Pop', 'Rock', 'Hip-Hop', 'Jazz', 'Classical', 'Electronic', 'Country', 'R&B']
    song_artists = [f'Artist_{i}' for i in range(1, 51)]
    
    songs = []
    for i in range(n_songs):
        genre = np.random.choice(song_genres)
        artist = np.random.choice(song_artists)
        duration = np.random.randint(120, 300)  # 2-5 minutes
        popularity = np.random.uniform(0, 1)
        
        songs.append({
            'song_id': i + 1,
            'title': f'Song_{i+1}',
            'artist': artist,
            'genre': genre,
            'duration': duration,
            'popularity': popularity
        })
    
    songs_df = pd.DataFrame(songs)
    
    # Generate listening history
    listens = []
    for _ in range(n_listens):
        user_id = np.random.randint(1, n_users + 1)
        song_id = np.random.randint(1, n_songs + 1)
        play_count = np.random.poisson(3) + 1  # At least 1 play
        
        listens.append({
            'user_id': user_id,
            'song_id': song_id,
            'play_count': play_count,
            'timestamp': np.random.randint(1000000000, 1600000000)
        })
    
    listens_df = pd.DataFrame(listens)
    
    return songs_df, listens_df

# Generate music data
songs_df, listens_df = generate_music_data()
print(f"Songs: {len(songs_df)}")
print(f"Listens: {len(listens_df)}")

# Create listening matrix
listening_matrix = listens_df.groupby(['user_id', 'song_id'])['play_count'].sum().unstack(fill_value=0)
print(f"Listening matrix shape: {listening_matrix.shape}")

# Hybrid Recommendation System
print("\n--- Hybrid Recommendation System ---")

def hybrid_recommendations(listening_matrix, songs_df, user_id, n_recommendations=5):
    """Hybrid recommendations combining collaborative and content-based filtering"""
    
    # Collaborative filtering (user-based)
    if user_id in listening_matrix.index:
        user_listens = listening_matrix.loc[user_id]
        similarities = cosine_similarity([user_listens], listening_matrix)[0]
        similar_users = np.argsort(similarities)[::-1][1:6]  # Top 5 similar users
        
        # Get songs from similar users
        cf_recommendations = set()
        for similar_user in similar_users:
            similar_user_id = listening_matrix.index[similar_user]
            similar_user_listens = listening_matrix.loc[similar_user_id]
            cf_recommendations.update(similar_user_listens[similar_user_listens > 0].index.tolist())
    else:
        cf_recommendations = set()
    
    # Content-based filtering (genre-based)
    if user_id in listening_matrix.index:
        user_listens = listening_matrix.loc[user_id]
        user_songs = user_listens[user_listens > 0].index.tolist()
        
        if user_songs:
            # Get user's preferred genres
            user_genres = songs_df.loc[songs_df['song_id'].isin(user_songs), 'genre'].value_counts()
            preferred_genres = user_genres.head(3).index.tolist()
            
            # Find songs in preferred genres
            cb_recommendations = set(songs_df[songs_df['genre'].isin(preferred_genres)]['song_id'].tolist())
        else:
            cb_recommendations = set()
    else:
        cb_recommendations = set()
    
    # Combine recommendations
    all_recommendations = cf_recommendations.union(cb_recommendations)
    
    # Remove already listened songs
    if user_id in listening_matrix.index:
        listened_songs = set(listening_matrix.loc[user_id][listening_matrix.loc[user_id] > 0].index.tolist())
        all_recommendations = all_recommendations - listened_songs
    
    # Score recommendations
    recommendation_scores = []
    for song_id in all_recommendations:
        score = 0
        
        # Collaborative filtering score
        if song_id in cf_recommendations:
            score += 2
        
        # Content-based score
        if song_id in cb_recommendations:
            score += 1
        
        # Popularity bonus
        song_popularity = songs_df.loc[song_id-1, 'popularity']
        score += song_popularity
        
        recommendation_scores.append((song_id, score))
    
    # Sort by score
    recommendation_scores.sort(key=lambda x: x[1], reverse=True)
    
    return recommendation_scores[:n_recommendations]

# Test hybrid recommendations
test_user_music = listening_matrix.index[0] if len(listening_matrix) > 0 else 1
hybrid_recommendations = hybrid_recommendations(listening_matrix, songs_df, test_user_music)
print(f"\nHybrid recommendations for user {test_user_music}:")
for song_id, score in hybrid_recommendations:
    song_title = songs_df.loc[song_id-1, 'title']
    song_artist = songs_df.loc[song_id-1, 'artist']
    song_genre = songs_df.loc[song_id-1, 'genre']
    print(f"  {song_title} by {song_artist} ({song_genre}): {score:.2f}")

# 4. Neural Network-Based Recommendations (if TensorFlow available)
if TENSORFLOW_AVAILABLE:
    print("\n=== Neural Network-Based Recommendations ===")
    
    def create_neural_recommendation_model(n_users, n_items, embedding_dim=50):
        """Create neural network recommendation model"""
        # User input
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(n_users, embedding_dim, name='user_embedding')(user_input)
        user_embedding = Flatten()(user_embedding)
        
        # Item input
        item_input = Input(shape=(1,), name='item_input')
        item_embedding = Embedding(n_items, embedding_dim, name='item_embedding')(item_input)
        item_embedding = Flatten()(item_embedding)
        
        # Concatenate and add dense layers
        concat = Concatenate()([user_embedding, item_embedding])
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        output = Dense(1, activation='linear')(dropout2)
        
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    # Prepare data for neural network
    def prepare_nn_data(ratings_df, n_users, n_items):
        """Prepare data for neural network training"""
        # Create user and item mappings
        user_mapping = {user_id: idx for idx, user_id in enumerate(range(1, n_users + 1))}
        item_mapping = {item_id: idx for idx, item_id in enumerate(range(1, n_items + 1))}
        
        # Convert to indices
        user_indices = [user_mapping[user_id] for user_id in ratings_df['user_id']]
        item_indices = [item_mapping[item_id] for item_id in ratings_df['movie_id']]
        ratings = ratings_df['rating'].values
        
        return np.array(user_indices), np.array(item_indices), np.array(ratings)
    
    # Prepare movie data for neural network
    user_indices, item_indices, ratings = prepare_nn_data(ratings_df, len(users_df), len(movies_df))
    
    # Split data
    X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
        user_indices, item_indices, ratings, test_size=0.2, random_state=42
    )
    
    # Create and train model
    nn_model = create_neural_recommendation_model(len(users_df), len(movies_df))
    print("Training neural network recommendation model...")
    history = nn_model.fit(
        [X_user_train, X_item_train], y_train,
        epochs=20, batch_size=64, validation_split=0.1, verbose=0
    )
    
    # Evaluate model
    y_pred = nn_model.predict([X_user_test, X_item_test]).flatten()
    nn_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    nn_mae = mean_absolute_error(y_test, y_pred)
    print(f"Neural Network RMSE: {nn_rmse:.3f}")
    print(f"Neural Network MAE: {nn_mae:.3f}")

# 5. Evaluation and Comparison
print("\n=== Evaluation and Comparison ===")

# Function to evaluate recommendation systems
def evaluate_recommendations(ratings_matrix, test_ratio=0.2):
    """Evaluate recommendation systems using train-test split"""
    # Create train-test split
    train_matrix = ratings_matrix.copy()
    test_ratings = []
    
    for user_id in ratings_matrix.index:
        user_ratings = ratings_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings.notna()].index.tolist()
        
        if len(rated_items) > 1:
            n_test = max(1, int(len(rated_items) * test_ratio))
            test_items = np.random.choice(rated_items, n_test, replace=False)
            
            for item_id in test_items:
                test_ratings.append((user_id, item_id, user_ratings[item_id]))
                train_matrix.loc[user_id, item_id] = np.nan
    
    return train_matrix, test_ratings

# Evaluate different methods
train_matrix, test_ratings = evaluate_recommendations(ratings_matrix)

# Calculate predictions for test set
predictions = []
for user_id, item_id, actual_rating in test_ratings:
    # SVD prediction
    svd_pred = predict_rating_svd(user_id, item_id, user_factors, movie_factors, train_matrix)
    predictions.append({
        'user_id': user_id,
        'item_id': item_id,
        'actual': actual_rating,
        'svd_pred': svd_pred
    })

predictions_df = pd.DataFrame(predictions)

# Calculate metrics
svd_rmse = np.sqrt(mean_squared_error(predictions_df['actual'], predictions_df['svd_pred']))
svd_mae = mean_absolute_error(predictions_df['actual'], predictions_df['svd_pred'])

print(f"Evaluation Results:")
print(f"SVD RMSE: {svd_rmse:.3f}")
print(f"SVD MAE: {svd_mae:.3f}")

if TENSORFLOW_AVAILABLE:
    print(f"Neural Network RMSE: {nn_rmse:.3f}")
    print(f"Neural Network MAE: {nn_mae:.3f}")

# 6. Visualization
print("\n=== Visualization ===")

# Create visualization plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Rating distribution
axes[0, 0].hist(ratings_df['rating'], bins=20, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Rating Distribution')

# 2. User activity distribution
user_activity = ratings_df.groupby('user_id').size()
axes[0, 1].hist(user_activity, bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Number of Ratings per User')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('User Activity Distribution')

# 3. Movie popularity distribution
movie_popularity = ratings_df.groupby('movie_id').size()
axes[0, 2].hist(movie_popularity, bins=30, alpha=0.7, edgecolor='black')
axes[0, 2].set_xlabel('Number of Ratings per Movie')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Movie Popularity Distribution')

# 4. Genre distribution
genre_counts = movies_df['genres'].str.split('|').explode().value_counts()
axes[1, 0].bar(genre_counts.index, genre_counts.values, alpha=0.7)
axes[1, 0].set_xlabel('Genre')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Genre Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. SVD explained variance
axes[1, 1].plot(np.cumsum(svd.explained_variance_ratio_), marker='o')
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Cumulative Explained Variance')
axes[1, 1].set_title('SVD Explained Variance')
axes[1, 1].grid(True, alpha=0.3)

# 6. Prediction vs Actual (if we have predictions)
if len(predictions_df) > 0:
    axes[1, 2].scatter(predictions_df['actual'], predictions_df['svd_pred'], alpha=0.6)
    axes[1, 2].plot([predictions_df['actual'].min(), predictions_df['actual'].max()], 
                    [predictions_df['actual'].min(), predictions_df['actual'].max()], 'r--', lw=2)
    axes[1, 2].set_xlabel('Actual Rating')
    axes[1, 2].set_ylabel('Predicted Rating')
    axes[1, 2].set_title('SVD: Predicted vs Actual Ratings')
    axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Summary and Recommendations
print("\n=== Summary and Recommendations ===")

print("Recommendation System Performance:")
print(f"1. User-Based Collaborative Filtering:")
print(f"   - Pros: Simple, interpretable, good for cold-start")
print(f"   - Cons: Scalability issues, sparsity problems")

print(f"\n2. Content-Based Filtering:")
print(f"   - Pros: No cold-start, interpretable, handles new items")
print(f"   - Cons: Limited by feature quality, overspecialization")

print(f"\n3. Matrix Factorization (SVD):")
print(f"   - Pros: Handles sparsity, captures latent factors")
print(f"   - Cons: Cold-start problem, less interpretable")

print(f"\n4. Non-negative Matrix Factorization:")
print(f"   - Pros: Non-negative factors, good for implicit feedback")
print(f"   - Cons: Cold-start problem, computational cost")

print(f"\n5. Hybrid Methods:")
print(f"   - Pros: Combines strengths of multiple approaches")
print(f"   - Cons: More complex, requires careful tuning")

if TENSORFLOW_AVAILABLE:
    print(f"\n6. Neural Networks:")
    print(f"   - Pros: Captures complex patterns, handles non-linear relationships")
    print(f"   - Cons: Requires more data, less interpretable, computational cost")

print(f"\nRecommendations:")
print(f"- Use collaborative filtering for established user bases")
print(f"- Use content-based filtering for new users/items")
print(f"- Use matrix factorization for large, sparse datasets")
print(f"- Use hybrid methods for best overall performance")
print(f"- Use neural networks for complex, non-linear patterns")
print(f"- Always consider cold-start and scalability issues") 