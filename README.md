# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

## Spotify Music Recommender

**Deadline**: Sunday, Feb 23th 11:59 pm PST <br>
**Author**: Matthew Budding
---

## Overview

Recommends 5 tracks to the user based off of the description they give of the type of music they like


1. **Dataset**  
   Used a modified version of the data set [Spotify Tracks Genre](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset/data) by taking 5 tracks from each genre in the dataset to create a smaller version to use in this recommender
   (Script to clean the data in the repository)

3. **Approach**  
   - Takes the genre of the track and terms like "valence" "speechiness" and adds them to a description of the song 
   - Terms like valence and speechiness are only added if the are at least 70% of the maxmimum value in their respective column to accurately describe each song
   - Then the description of every song is compared to the user input using tf-idf and cosine similarity to find songs best matching the description the user gave.
   - Return the top 5 tracks that match the description the user gave

4. **Summary & Instructions**  
   - A short `README.md` that includes:
     - **Setup**: Using python version 3.9.5, `pip install -r requirements.txt` to download necessary packages (pandas and scikit-learn)
     - **Running**: `python ./src/main.py` (takes about 1-2 seconds to load)
     - **Results**: Will output 5 tracks most similar to the description given: ![image](https://github.com/user-attachments/assets/5f124d65-adda-4640-b442-5196c0f2c2b7)

5. **Salary Expected per Month**<br>
    - Minimum salary offered on handshake: $20/hr <br>
    - Assumed 20 hours per week as stated on handshake: **$1733 per month**

