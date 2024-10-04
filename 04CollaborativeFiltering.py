# Import necessary libraries
import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy
import joblib
import pickle

# Load the balanced interactions dataset
df = pd.read_csv('balanced_interactions.csv')

# Subset with necessary columns: 'customer_id', 'article_id', and 'interaction'
df_subset = df[['customer_id', 'article_id', 'interaction']].copy()

# Prepare the data for the Surprise library
reader = Reader(rating_scale=(0, 1))  # Assuming interactions are binary (0 or 1)
data = Dataset.load_from_df(df_subset[['customer_id', 'article_id', 'interaction']], reader)

# Train-Test split: Ensure the trainset and testset split is randomized each time
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Use SVD (Singular Value Decomposition) for collaborative filtering
model = SVD(n_factors=20, n_epochs=10, lr_all=0.002, reg_all=0.1)

# Train the model on the trainset
model.fit(trainset)

# Test the model on the testset
predictions = model.test(testset)

# Check the accuracy of the model on the testset
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f'RMSE on test set: {rmse}')
print(f'MAE on test set: {mae}')

# Cross-validate the model (5-fold cross-validation on full dataset)
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Save the trained model for later use
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=4)

# # Save the trained model for later use
# joblib.dump(model, 'svd_model.pkl', compress=True)


# # Example of making a prediction for a specific customer and article
# customer_id = '46a17af908b3e2a735a08156cfb07ce782c06db2d8c17919190951be79a85441'
# article_id = '736049001'
# prediction = model.predict(customer_id, article_id)
# print(f'Predicted interaction: {prediction.est}')  
