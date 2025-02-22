import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
import scipy.sparse as sp

# -------------------------------
# 1. Data Loading and Filtering
# -------------------------------

# Paths to processed data files
interactions_path = '/content/drive/MyDrive/dataset/processed_interactions.csv'
metadata_path = '/content/drive/MyDrive/dataset/processed_metadata.csv'

# Load the processed data
interactions_df = pd.read_csv(interactions_path)
metadata_df = pd.read_csv(metadata_path)

# Filter users with at least 3 interactions to ensure they appear in both train and test sets
user_interaction_counts = interactions_df.groupby('user_id').size().reset_index(name='count')
eligible_users = user_interaction_counts[user_interaction_counts['count'] >= 3]

# -------------------------------
# 2. Stratified Sampling of Users
# -------------------------------
# We'll stratify the eligible users by their interaction counts.
# Use qcut to create quantile bins (e.g., 4 bins) to preserve the overall distribution.
eligible_users['bin'] = pd.qcut(eligible_users['count'], q=4, duplicates='drop')

# Define the total number of users to sample (roughly estimated to yield ~10,000 interactions overall)
# (For example, if average interactions per user ~10, sampling 1000 users should be about right.)
total_users_to_sample = 1000

sampled_users_list = []
for grp, group_df in eligible_users.groupby('bin'):
    # Calculate the number to sample from this bin proportionally
    n_in_bin = len(group_df)
    n_to_sample = int(round(n_in_bin / len(eligible_users) * total_users_to_sample))
    # Ensure at least one user is sampled from each bin
    n_to_sample = max(1, n_to_sample)
    sampled_users_list.append(group_df.sample(n=n_to_sample, random_state=42))

sampled_users = pd.concat(sampled_users_list)
sampled_user_ids = sampled_users['user_id'].unique()

# Get all interactions for the sampled users
interactions_sample = interactions_df[interactions_df['user_id'].isin(sampled_user_ids)]

# -------------------------------
# 3. Metadata Sampling
# -------------------------------

# From metadata, select only the stories present in the sampled interactions
sampled_stories = interactions_sample['pratilipi_id'].unique()
metadata_sample = metadata_df[metadata_df['pratilipi_id'].isin(sampled_stories)]

# Additionally, sample 10,000 extra random stories from metadata (to widen the item pool)
remaining_stories = metadata_df[~metadata_df['pratilipi_id'].isin(sampled_stories)]
additional_stories = remaining_stories.sample(n=10000, random_state=42)
metadata_sample = pd.concat([metadata_sample, additional_stories]).drop_duplicates(subset='pratilipi_id')

# Ensure that the interactions sample only contains pratilipi_ids available in metadata_sample
interactions_sample = interactions_sample[interactions_sample['pratilipi_id'].isin(metadata_sample['pratilipi_id'])]

# -------------------------------
# 4. Build the LightFM Dataset
# -------------------------------

# Initialize the LightFM Dataset with users, items, and item features (using 'category_name')
dataset = Dataset()
dataset.fit(
    users=interactions_sample['user_id'].unique(),
    items=metadata_sample['pratilipi_id'].unique(),
    item_features=metadata_sample['category_name'].unique()
)

# Build the interactions matrix from the interactions_sample dataframe
(interactions, _) = dataset.build_interactions(
    ((row['user_id'], row['pratilipi_id']) for _, row in interactions_sample.iterrows())
)

# Build item features matrix using the 'category_name' field from metadata_sample
item_features = dataset.build_item_features(
    ((row['pratilipi_id'], [row['category_name']]) for _, row in metadata_sample.iterrows())
)

# -------------------------------
# 5. Split Data and Train the Model
# -------------------------------

# Split the interactions matrix into 75% training and 25% test sets
train, test = random_train_test_split(interactions, test_percentage=0.25, random_state=np.random.RandomState(42))

# Initialize the LightFM model using the WARP loss (as originally used)
model = LightFM(loss='warp', learning_rate=0.01)
model.fit(train, item_features=item_features, epochs=50, num_threads=4)

# -------------------------------
# 6. Evaluation Metrics Calculation
# -------------------------------

# Compute evaluation metrics on the test set using LightFM's evaluation functions
prec_at_5 = precision_at_k(model, test, k=5, item_features=item_features).mean()
rec_at_5 = recall_at_k(model, test, k=5, item_features=item_features).mean()
auc = auc_score(model, test, item_features=item_features).mean()

# Create a DataFrame with the evaluation results
metrics_df = pd.DataFrame({
    'precision_at_5': [prec_at_5],
    'recall_at_5': [rec_at_5],
    'auc_score': [auc]
})

# Save the evaluation metrics to a CSV file
metrics_df.to_csv('test_metrics.csv', index=False)

print("Evaluation Metrics on Test Set:")
'''print(f"Precision@5: {prec_at_5}")
print(f"Recall@5: {rec_at_5}")'''
print(f"AUC Score: {auc}")

# -------------------------------
# 7. Generate and Save Recommendations
# -------------------------------

# Retrieve the mapping dictionaries for users and items from the dataset
user_id_map, _, item_id_map, _ = dataset.mapping()
# Create a reverse mapping for items (from internal ID to original pratilipi_id)
item_reverse_mapping = {v: k for k, v in item_id_map.items()}

# Get the number of items from the interactions matrix shape
n_users, n_items = interactions.shape

# Generate top-5 recommendations for every user in the dataset
recommendations = []
for user in user_id_map:
    internal_user_id = user_id_map[user]
    # Predict scores for all items for the given user
    scores = model.predict(internal_user_id, np.arange(n_items), item_features=item_features)
    # Get indices of the top-5 items (highest predicted scores)
    top_items = np.argsort(-scores)[:5]
    # Convert internal item IDs back to original pratilipi_ids
    recs = [item_reverse_mapping[i] for i in top_items]
    recommendations.append({
        'user_id': user,
        'recommended_pratilipi_ids': recs
    })

# Convert the recommendations list into a DataFrame.
# The recommended pratilipi IDs are saved as a comma-separated string.
recs_df = pd.DataFrame(recommendations)
recs_df['recommended_pratilipi_ids'] = recs_df['recommended_pratilipi_ids'].apply(lambda x: ','.join(map(str, x)))

# Save the recommendations to a CSV file
recs_df.to_csv('test_recommendations.csv', index=False)

print("Recommendations for users have been saved to 'test_recommendations.csv'.")
print("Evaluation metrics have been saved to 'test_metrics.csv'.")
