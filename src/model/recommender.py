import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MusicRecommender:
    def __init__(self, data_path):
        # Load the dataset
        self.data = pd.read_csv(data_path)

        
        self.data = self.data.fillna('')

        
        # Calculate the maximum values for each column
        self.max_values = self.data[['popularity', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']].max()
        
        # Preprocess the data by combining relevant columns into a single description
        def create_description(row):
            description = f"{row['track_genre']} {row['popularity']} {row['loudness']} {row['speechiness']} {row['acousticness']} {row['instrumentalness']} {row['liveness']} {row['valence']}"
            for col in ['popularity', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']:
                if row[col] >= 0.7 * self.max_values[col]:  # Adds the column to the description if the value for the track is 70% or more of the highest value in the column
                    description += f" {col}"
            
            return description
        
        self.data['description'] = self.data.apply(create_description, axis=1)

        
        # Initialize the TF-IDF Vectorizer with adjusted parameters
        self.tfidf = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2)
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['description'])


    def recommend_music(self, user_input, top_n=5):

        
        # Vectorize the user input
        user_input_tfidf = self.tfidf.transform([user_input])

        
        # Compute the cosine similarity between the user input and the descriptions
        cosine_sim = linear_kernel(user_input_tfidf, self.tfidf_matrix)

        
        # Get the similarity scores for the user input
        sim_scores = list(enumerate(cosine_sim[0]))
        
        # Sort the scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        
        # Get the indices of the top_n most similar items
        sim_scores = sim_scores[:top_n]
        music_indices = [i[0] for i in sim_scores]
        print(f"Top {top_n} most similar items indices: {music_indices}")
        
        # Return the top_n most similar items
        recommended_music = self.data.iloc[music_indices]
        
        return recommended_music