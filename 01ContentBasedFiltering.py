from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import pickle

# Load and sample the dataset
df = pd.read_csv('processed_data.csv')

# reduce the size of the dataset to 10,000 samples from 20,000
df = df.sample(n=10000, random_state=1) 

df.to_csv('processed_data_sample.csv', index=False)

# Reset index to ensure it starts from 0
df.reset_index(drop=True, inplace=True)

# Combine important product features along with the season label
df['combined_features'] = (df['product_type_name'].fillna('') + " " +
                           df['garment_group_name'].fillna('') + " " +
                           df['product_group_name'].fillna('') + " " +
                           df['season_label'].fillna(''))

# Create a TfidfVectorizer to convert text data into feature vectors
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features into TF-IDF vectors
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

# Calculate cosine similarity between all items based on the combined features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the TfidfVectorizer to a file
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f, protocol=4)

# Save cosine similarity matrix
with open('cosine_similarity.pkl', 'wb') as f:
    pickle.dump(cosine_sim, f, protocol=4)

# # Save the TfidfVectorizer to a file
# joblib.dump(tfidf, 'tfidf_vectorizer.pkl', compress=True)

# # Save with compression
# joblib.dump(cosine_sim, 'cosine_similarity.pkl', compress=True)





# # Example usage
# article_id = df['article_id'].sample(n=1).values[0]
# print(f"Example article ID: {article_id}")
# similar_articles = get_similar_articles(article_id, cosine_sim, df)
# print(f"Articles similar to {article_id}: {similar_articles}")
