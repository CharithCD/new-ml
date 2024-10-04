import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

# Load the balanced interactions dataset
df = pd.read_csv('balanced_interactions.csv')

# Subset with necessary columns: 'customer_id', 'article_id', and 'interaction'
# 'interaction' column already represents 1 for interaction and 0 for no interaction
df_subset = df[['customer_id', 'article_id', 'interaction']].copy()

# Prepare the data for Surprise library
reader = Reader(rating_scale=(0, 1))  # Assuming interactions are binary (0 or 1)
data = Dataset.load_from_df(df_subset[['customer_id', 'article_id', 'interaction']], reader)

# Define a parameter grid for GridSearchCV
param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs': [10, 20, 30],
    'lr_all': [0.002, 0.005, 0.01],
    'reg_all': [0.02, 0.1, 0.4]
}

# Initialize GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
grid_search.fit(data)

# Output the best parameters and best scores
print(f'Best RMSE: {grid_search.best_score["rmse"]}')
print(f'Best MAE: {grid_search.best_score["mae"]}')
print('Best parameters:', grid_search.best_params['rmse'])


# output:
# Best RMSE: 0.5046329170703882
# Best MAE: 0.5015504547891212
# Best parameters: {'n_factors': 20, 'n_epochs': 10, 'lr_all': 0.002, 'reg_all': 0.1}
