import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MusicRecommender:
    def __init__(self, data_path):
        # Load the dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data.fillna('')
        # Preprocess the data by combining relevant columns into a single description
        self.data['description'] = (
            self.data['track_name'] + ' ' +
            self.data['artists'] + ' ' +
            self.data['track_genre'] + ' ' +
            self.data['popularity'].astype(str) + ' ' +
            self.data['loudness'].astype(str) + ' ' +
            self.data['speechiness'].astype(str) + ' ' +
            self.data['acousticness'].astype(str) + ' '+ 
            self.data['instrumentalness'].astype(str) + ' ' +
            self.data['liveness'].astype(str) + ' ' +
            self.data['valence'].astype(str) 
        )
        # Initialize the TF-IDF Vectorizer
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['description'])

    def recommend_music(self, user_input, top_n=10):
        # Combine user input with the dataset
        data_with_input = self.data.copy()
        data_with_input.loc[len(data_with_input)] = [None] * (len(self.data.columns) - 1) + [user_input]

        # Vectorize the text data including user input
        tfidf_matrix_with_input = self.tfidf.fit_transform(data_with_input['description'])

        # Compute the cosine similarity matrix
        cosine_sim = linear_kernel(tfidf_matrix_with_input, tfidf_matrix_with_input)

        # Get the similarity scores for the user input
        sim_scores = list(enumerate(cosine_sim[-1]))

        # Sort the scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top_n most similar items
        sim_scores = sim_scores[1:top_n+1]
        music_indices = [i[0] for i in sim_scores]

        # Return the top_n most similar items
        return self.data.iloc[music_indices]