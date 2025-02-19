import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
user_artists = pd.read_csv('./dataset/user_artists.dat', delimiter='\t')
artists = pd.read_csv('./dataset/artists.dat', delimiter='\t')

# Create a user-item interaction matrix
user_item_matrix = user_artists.pivot(index='userID', columns='artistID', values='weight').fillna(0)

# Compute the cosine similarity matrix
artist_similarity = cosine_similarity(user_item_matrix.T)
artist_similarity_df = pd.DataFrame(artist_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

def get_artist_recommendations(artist_name, n_recommendations=5):
    # Check if the artist exists in the dataset
    if not artists['name'].str.contains(artist_name).any():
        raise ValueError(f"Artist '{artist_name}' not found in the dataset.")
    
    # Find the artist ID
    artist_id = artists[artists['name'] == artist_name]['id'].values[0]
    
    # Check if the artist ID exists in the similarity DataFrame, if it doesn't that means we dont have anyone or enough people who listens to this artist in the dataset
    if artist_id not in artist_similarity_df.columns:
        raise KeyError(f"Artist {artist_name} (ID: '{artist_id}') not found in the similarity DataFrame.")
    
    # Get the similarity scores for the artist
    similar_artists = artist_similarity_df[artist_id].sort_values(ascending=False)
    
    # Get the top n recommendations
    top_recommendations = similar_artists.head(n_recommendations + 1).index[1:]
    
    # Get the artist names
    recommended_artists = artists[artists['id'].isin(top_recommendations)]['name']
    
    return recommended_artists

# Main function to get user input and print recommendations
def main():
    artist_name = input("Enter an artist you like: ")
    try:
        recommendations = get_artist_recommendations(artist_name)
        if recommendations.empty:
            print(f"No recommendations found for artist '{artist_name}'.")
        else:
            print(f"Recommendations for artist '{artist_name}':")
            for artist in recommendations:
                print(artist)
    except ValueError as e:
        print(e)
    except KeyError as e:
        print(e)

if __name__ == "__main__":
    main()