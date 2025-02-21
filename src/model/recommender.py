import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MusicRecommender:
    def __init__(self, data_path):
        # Load the dataset
        self.data = pd.read_csv(data_path)
        print("Dataset loaded.")
        
        self.data = self.data.fillna('')
        print("NaN values filled.")
        
        # Calculate the maximum values for each column
        self.max_values = self.data[['popularity', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']].max()
        
        # Preprocess the data by combining relevant columns into a single description
        def create_description(row):
            description = f"{row['track_genre']} {row['popularity']} {row['loudness']} {row['speechiness']} {row['acousticness']} {row['instrumentalness']} {row['liveness']} {row['valence']}"
            for col in ['popularity', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']:
                if row[col] >= 0.7 * self.max_values[col]:  # Adjust the range as needed
                    description += f" {col}"
            
            return description
        
        self.data['description'] = self.data.apply(create_description, axis=1)
        print("Data preprocessed.")
        
        # Initialize the TF-IDF Vectorizer with adjusted parameters
        self.tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['description'])
        print("TF-IDF matrix created.")

    def recommend_music(self, user_input, top_n=5):
        print(f"User input: {user_input}")
        
        # Vectorize the user input
        user_input_tfidf = self.tfidf.transform([user_input])
        print("User input vectorized.")
        
        # Compute the cosine similarity between the user input and the descriptions
        cosine_sim = linear_kernel(user_input_tfidf, self.tfidf_matrix)
        print("Cosine similarity matrix computed.")
        
        # Get the similarity scores for the user input
        sim_scores = list(enumerate(cosine_sim[0]))
        print("Similarity scores computed.")
        
        # Sort the scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        print("Similarity scores sorted.")
        
        # Get the indices of the top_n most similar items
        sim_scores = sim_scores[:top_n]
        music_indices = [i[0] for i in sim_scores]
        print(f"Top {top_n} most similar items indices: {music_indices}")
        
        # Return the top_n most similar items
        recommended_music = self.data.iloc[music_indices]
        
        return recommended_music