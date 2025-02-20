from model.recommender import MusicRecommender

# Initialize the recommender with the path to the dataset
recommender = MusicRecommender('./src/dataset/train_reduced.csv')

# Get user input
user_input = input("Enter a description of the type of music you like: ")

# Get recommendations based on user input
recommended_music = recommender.recommend_music(user_input)

recommended_music = recommended_music[['track_name', 'artists', 'track_genre']]
# Print the recommended music
print(recommended_music)