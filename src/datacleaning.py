import pandas as pd

# Load the dataset
file_path = './src/dataset/train.csv'
df = pd.read_csv(file_path)

# Function to keep only 5 rows per genre
def keep_top_n_per_genre(df, n=5):
    return df.groupby('track_genre').head(n)

# Apply the function
df_reduced = keep_top_n_per_genre(df)

# Save the reduced dataset to a new CSV file
output_file_path = './src/dataset/train_reduced.csv'
df_reduced.to_csv(output_file_path, index=False)

print(f"Reduced dataset saved to {output_file_path}")