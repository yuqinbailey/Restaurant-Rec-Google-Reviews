import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_similar_users_avg_rating(user_df, df_filtered, user, restaurant, embedding='bert_embedding', k=10):
    # Filter df_filtered for reviews on the specific restaurant
    filtered_reviews = df_filtered[df_filtered['gmap_id'] == restaurant]
    
    # Get the BERT embedding of the target user from train_df
    user_df['user_id'] = user_df['user_id'].astype(str)
    # Check if the user is in the DataFrame and if so, retrieve the embedding
    if user in user_df['user_id'].values:
        target_embedding = user_df[user_df['user_id'] == user][embedding].values[0]
    else:
        return None
        
    # Retrieve embeddings for all users who have commented on this restaurant
    user_embeddings = user_df[user_df['user_id'].isin(filtered_reviews['user_id'])]
    if len(user_embeddings) == 0:
        return None
    
    # Calculate cosine similarity between target user and all users in user_embeddings
    similarities = cosine_similarity([target_embedding], np.stack(user_embeddings[embedding].values))
    
    # Create a DataFrame for similarities
    similarity_df = pd.DataFrame({
        'user_id': user_embeddings['user_id'],
        'similarity': similarities.flatten()
    })
    
    # Sort by similarity and select the top 10
    top_similar_users = similarity_df.sort_values(by='similarity', ascending=False).head(k)
    
    # Merge to get ratings of these top similar users for the specific gmap_id
    top_user_ratings = top_similar_users.merge(filtered_reviews[['user_id', 'rating']], on='user_id', how='left')
    
    # Calculate the average rating
    average_rating = top_user_ratings['rating'].mean()
    
    return average_rating

def merge_df(df_filtered, num_comments=50):
    df = pd.read_csv("data/data_ma.csv")

    df_cleaned = pd.merge(df_filtered[df_filtered['comment_count'] >= num_comments], 
                        df[['user_id', 'name_y', 'gmap_id', 'latitude', 'longitude', 'num_of_reviews', 'price', 'avg_rating']], 
                        on=['user_id', 'gmap_id'],
                        how='left')
    df_cleaned['user_id'] = df_cleaned['user_id'].astype(str)
    return df_cleaned