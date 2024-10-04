import pandas as pd
import random
from tqdm import tqdm

# Load the processed data (transactions)
df = pd.read_csv('processed_data_sample.csv.csv')

# Unique customers and articles
customers = df['customer_id'].unique()
articles = df['article_id'].unique()

# Create a DataFrame to store the interactions (positive and negative)
interaction_data = []

# Iterate over each customer with a progress bar
for customer in tqdm(customers, desc="Processing customers", unit="customer"):
    # Get the articles that the customer has purchased
    purchased_articles = df[df['customer_id'] == customer]['article_id'].unique()
    
    # Add positive interactions (purchases) as 1
    for article in purchased_articles:
        interaction_data.append([customer, article, 1])
    
    # Randomly sample articles that the customer did NOT purchase
    non_purchased_articles = list(set(articles) - set(purchased_articles))
    
    # Add some negative interactions (non-purchases) as 0 (You can adjust the number of non-purchases to balance the data)
    sampled_articles = random.sample(non_purchased_articles, min(len(purchased_articles), len(non_purchased_articles)))
    
    for article in sampled_articles:
        interaction_data.append([customer, article, 0])

# Convert the interaction data into a DataFrame
interaction_df = pd.DataFrame(interaction_data, columns=['customer_id', 'article_id', 'interaction'])

# Save the balanced interaction data to a CSV
interaction_df.to_csv('balanced_interactions.csv', index=False)

print("Balanced interaction CSV file created successfully!")
