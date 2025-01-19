import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the dataset from a CSV file into a DataFrame
new_df = pd.read_csv('export2.csv')

# Vectorization: Convert the 'tags' column into a numerical feature matrix
# Use CountVectorizer with a maximum of 5000 features and English stopwords removal
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()  # Convert to a dense array

# Debugging: Optional exploration of the feature matrix and vocabulary
count = 0
# Uncomment the following block to see the vectors for the first 10 rows
# for i in vector:
#     print(count, i)
#     count += 1
#     if count == 10:
#         break

# Uncomment the following block to see the first 10 feature names
# for i in cv.get_feature_names_out():
#     print(count, i)
#     count += 1
#     if count == 10:
#         break

# Compute cosine similarity for the feature matrix
similarity = cosine_similarity(vector)

# Uncomment the following lines to debug and inspect similarity values
# print(similarity.shape)  # Shape of the similarity matrix
# print(similarity[0])     # Similarity scores for the first movie
# print(similarity[1])     # Similarity scores for the second movie

# Recommendation function
def recommend(movie):
    # Convert input movie name to title case to ensure it matches the dataset format
    movie = movie.title()
    try:
        # Find the index of the given movie title in the DataFrame
        movie_index = new_df[new_df['title'] == movie].index[0]
        # Get the similarity scores for the movie
        distances = similarity[movie_index]
        # Sort the movies based on similarity scores in descending order
        # Take the top 6 most similar movies
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[0:6]

        # Print the recommended movies
        print("Movies you may like:")
        for i in movies_list:
            print(new_df.iloc[i[0]].title)
    except IndexError:
        # Handle the case when the movie is not found in the dataset
        print("Movie not found in database")

# Take movie input from the user
name = input("Enter movie name: ")
recommend(name)
