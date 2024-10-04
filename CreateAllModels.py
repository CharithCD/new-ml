import pandas as pd
import pickle
import gzip
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm
import joblib

def create_models():
    print("Loading and processing data...")
    # Load the full dataset
    df = pd.read_csv('processed_data.csv')
    
    # Sample the data
    df = df.sample(n=10000, random_state=1)
    df.reset_index(drop=True, inplace=True)
    
    # Save the sampled data for future use
    df.to_csv('processed_data_sample.csv', index=False)
    print("Sampled data saved to processed_data_sample.csv")

    # Content-based filtering
    print("Creating content-based model...")
    df['combined_features'] = (df['product_type_name'].fillna('') + " " +
                               df['garment_group_name'].fillna('') + " " +
                               df['product_group_name'].fillna('') + " " +
                               df['season_label'].fillna(''))

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # # Save content-based model
    # print("Saving content-based model...")
    # with gzip.open('tfidf_vectorizer.pkl.gz', 'wb') as f:
    #     pickle.dump(tfidf, f, protocol=4)
    # with gzip.open('cosine_similarity.pkl.gz', 'wb') as f:
    #     pickle.dump(cosine_sim, f, protocol=4)

    # Save the TfidfVectorizer to a file
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl', compress=True)

    # Save with compression
    joblib.dump(cosine_sim, 'cosine_similarity.pkl', compress=True)

    # # Load the processed data (transactions)
    # df = pd.read_csv('processed_data_sample.csv')

    # # Unique customers and articles
    # customers = df['customer_id'].unique()
    # articles = df['article_id'].unique()

    # # Create a DataFrame to store the interactions (positive and negative)
    # interaction_data = []

    # # Iterate over each customer with a progress bar
    # for customer in tqdm(customers, desc="Processing customers", unit="customer"):
    #     # Get the articles that the customer has purchased
    #     purchased_articles = df[df['customer_id'] == customer]['article_id'].unique()

    #     # Add positive interactions (purchases) as 1
    #     for article in purchased_articles:
    #         interaction_data.append([customer, article, 1])

    #     # Randomly sample articles that the customer did NOT purchase
    #     non_purchased_articles = list(set(articles) - set(purchased_articles))

    #     # Add some negative interactions (non-purchases) as 0 (You can adjust the number of non-purchases to balance the data)
    #     sampled_articles = random.sample(non_purchased_articles, min(len(purchased_articles), len(non_purchased_articles)))

    #     for article in sampled_articles:
    #         interaction_data.append([customer, article, 0])

    # # Convert the interaction data into a DataFrame
    # interaction_df = pd.DataFrame(interaction_data, columns=['customer_id', 'article_id', 'interaction'])

    # # Save the balanced interaction data to a CSV
    # interaction_df.to_csv('balanced_interactions.csv', index=False)

    # print("Balanced interaction CSV file created successfully!")

    # # Load the balanced interactions dataset
    # print("Loading balanced interactions...")
    # df_interactions = pd.read_csv('balanced_interactions.csv')

    # # Filter the interactions to only include items from our sample
    # df_interactions = df_interactions[df_interactions['article_id'].isin(df['article_id'])]

    # # Collaborative filtering
    # print("Creating collaborative filtering model...")
    # reader = Reader(rating_scale=(0, 1))
    # data = Dataset.load_from_df(df_interactions[['customer_id', 'article_id', 'interaction']], reader)

    # model = SVD(n_factors=20, n_epochs=10, lr_all=0.002, reg_all=0.1)
    # trainset = data.build_full_trainset()
    # model.fit(trainset)

    # # # Save collaborative filtering model
    # # print("Saving collaborative filtering model...")
    # # with gzip.open('svd_model.pkl.gz', 'wb') as f:
    # #     pickle.dump(model, f, protocol=4)

    # # Save the trained model for later use
    # print("Saving collaborative filtering model...")
    # joblib.dump(model, 'svd_model.pkl', compress=True)

    # print("All models created and saved successfully!")

if __name__ == "__main__":
    create_models()