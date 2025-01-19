import numpy as np
import pandas as pd
from ast import literal_eval
from nltk.stem.porter import PorterStemmer

# Function to extract the 'name' field from the given JSON-like object (list of dictionaries)
def convert(obj):
    list1 = []
    for i in literal_eval(obj):  # Convert string representation of list into actual list
        list1.append(i['name'])
    return list1

# Function to extract the 'name' field but limits to the first 3 entries
def convert3(obj):
    list2 = []
    j = 0
    for i in literal_eval(obj):
        if j < 3:  # Restrict to 3 entries
            list2.append(i['name'])
            j += 1
        else:
            break
    return list2

# Function to extract the 'Director' from the crew field
def getdirector(obj):
    list3 = []
    for i in literal_eval(obj):
        if i['job'] == 'Director':  # Check if the job is 'Director'
            list3.append(i['name'])
            break  # Only one director is considered
    return list3

# Load the movies and credits datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the movies and credits datasets on the 'title' column
movies = movies.merge(credits, on='title')

# Select relevant columns for preprocessing
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# Check for duplicates and remove if any (no duplicates here based on your code)
print(movies.duplicated().sum())

# Preprocess the 'genres' and 'keywords' columns using the `convert` function
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Preprocess the 'cast' column using the `convert3` function to restrict to the first 3 names
movies['cast'] = movies['cast'].apply(convert3)

# Preprocess the 'crew' column to extract the director using the `getdirector` function
movies['crew'] = movies['crew'].apply(getdirector)

# Tokenize the 'overview' column into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from elements in lists in 'genres', 'cast', 'crew', and 'keywords'
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(' ', '') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(' ', '') for i in x])

# Create a new 'tags' column by combining 'overview', 'genres', 'keywords', 'cast', and 'crew'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with only the 'movie_id', 'title', and 'tags' columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert the 'tags' column from list to a single string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert the 'tags' column to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Display the first 20 rows of the processed DataFrame
print(new_df.head(20))

# Initialize the PorterStemmer for stemming
ps = PorterStemmer()

# Function to stem words in the 'tags' column
def stem(text):
    y = []
    for i in text.split():  # Split the string into words
        y.append(ps.stem(i))  # Stem each word
    return " ".join(y)  # Join the stemmed words back into a string

# Apply stemming to the 'tags' column
new_df['tags'] = new_df['tags'].apply(stem)

# Optionally save the preprocessed data to a CSV file
# new_df.to_csv('export2.csv', index=False)

print("Preprocessing complete.")
