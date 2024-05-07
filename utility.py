import pandas as pd
import numpy as np
import re
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity


def is_english(text):
    try: 
        return detect(text) == 'en'
    except: 
        return False

def get_emoji_pattern():
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    return emoji_pattern

def get_similar_users_avg_rating(user_df, df_filtered, user, restaurant, embedding_path, k=10):
    embeddings = np.load(embedding_path)

    # Filter df_filtered for reviews on the specific restaurant
    filtered_reviews = df_filtered[df_filtered['gmap_id'] == restaurant]
    
    # Get the BERT embedding of the target user from train_df
    user_df['user_id'] = user_df['user_id'].astype(str)
    # Check if the user is in the DataFrame and if so, retrieve the embedding
    if user in user_df['user_id'].values:
        target_index = user_df.index[user_df['user_id'] == user].tolist()[0]
        target_embedding = embeddings[target_index]
    else:
        return None
        
    # Retrieve embeddings for all users who have commented on this restaurant
    user_indices = user_df.index[user_df['user_id'].isin(filtered_reviews['user_id'])].tolist()
    if len(user_indices) == 0:
        return None
    
    # Calculate cosine similarity between target user and all users in user_embeddings
    similarities = cosine_similarity([target_embedding], embeddings[user_indices])
    
    # Create a DataFrame for similarities
    similarity_df = pd.DataFrame({
        'user_id': user_df.iloc[user_indices]['user_id'],
        'similarity': similarities.flatten()
    })
    
    # Sort by similarity and select the top 10
    top_similar_users = similarity_df.sort_values(by='similarity', ascending=False).head(k)
    
    # Merge to get ratings of these top similar users for the specific gmap_id
    top_user_ratings = top_similar_users.merge(filtered_reviews[['user_id', 'rating']], on='user_id', how='left')
    
    # Calculate the average rating
    average_rating = top_user_ratings['rating'].mean()
    
    return average_rating

def merge_df(df_filtered, num_comments=20):
    df = pd.read_csv("data/data_ma.csv")

    df_cleaned = pd.merge(df_filtered[df_filtered['comment_count'] >= num_comments], 
                        df[['user_id', 'name_y', 'gmap_id', 'latitude', 'longitude', 'num_of_reviews', 'price', 'avg_rating']], 
                        on=['user_id', 'gmap_id'],
                        how='left')
    df_cleaned['user_id'] = df_cleaned['user_id'].astype(str)
    return df_cleaned
