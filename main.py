import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
user_artists = pd.read_csv('./dataset/user_artists.dat', delimiter='\t')
artists = pd.read_csv('./dataset/artists.dat', delimiter='\t')

# Create a user-item interaction matrix
user_item_matrix = user_artists.pivot(index='userID', columns='artistID', values='weight').fillna(0)

# Compute the cosine similarity matrix
artist_similarity = cosine_similarity(user_item_matrix.T)
artist_similarity_df = pd.DataFrame(artist_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer().fit(artists['name'])

def find_artist_in_sentence(sentence):
    # Transform the input sentence to TF-IDF
    sentence_tfidf = tfidf_vectorizer.transform([sentence])
    
    # Compute cosine similarity between the input sentence and artist names
    cosine_similarities = cosine_similarity(sentence_tfidf, tfidf_vectorizer.transform(artists['name']))
    
    # Find the artist with the highest similarity score
    most_similar_artist_index = cosine_similarities.argmax()
    most_similar_artist = artists.iloc[most_similar_artist_index]['name']
    
    return most_similar_artist

def get_artist_recommendations(artist_names, n_recommendations=5):
    # Initialize a dictionary to store the aggregated similarity scores
    aggregated_similarities = pd.Series(0, index=artist_similarity_df.index)
    
    for artist_name in artist_names:
        # Check if the artist exists in the dataset
        if not artists['name'].str.contains(artist_name).any():
            raise ValueError(f"Artist '{artist_name}' not found in the dataset.")
        
        # Find the artist ID
        artist_id = artists[artists['name'] == artist_name]['id'].values[0]
        
        # Check if the artist ID exists in the similarity DataFrame
        if artist_id not in artist_similarity_df.columns:
            raise KeyError(f"Artist {artist_name} (ID: '{artist_id}') not found in the similarity DataFrame.")
        
        # Get the similarity scores for the artist and add to the aggregated similarities
        aggregated_similarities += artist_similarity_df[artist_id]
    
    # Sort the aggregated similarities and get the top n recommendations
    top_recommendations = aggregated_similarities.sort_values(ascending=False).head(n_recommendations + len(artist_names)).index
    
    # Filter out the input artists from the recommendations
    input_artist_ids = artists[artists['name'].isin(artist_names)]['id'].values
    top_recommendations = [artist_id for artist_id in top_recommendations if artist_id not in input_artist_ids][:n_recommendations]
    
    # Get the artist names
    recommended_artists = artists[artists['id'].isin(top_recommendations)]['name']
    
    return recommended_artists

# Main function to get user input and print recommendations
def main():
    sentences = input("Enter sentences with artists you like (separated by commas): ").split(',')
    try:
        artist_names = [find_artist_in_sentence(sentence.strip()) for sentence in sentences]
        print(f"Detected artists: {', '.join(artist_names)}")
        recommendations = get_artist_recommendations(artist_names)
        if recommendations.empty:
            print(f"No recommendations found for artists '{', '.join(artist_names)}'.")
        else:
            print(f"Recommendations for artists '{', '.join(artist_names)}':")
            for artist in recommendations:
                print(artist)
    except ValueError as e:
        print(e)
    except KeyError as e:
        print(e)

if __name__ == "__main__":
    main()