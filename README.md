# Movie Recommendation System

This project demonstrates a simple movie recommendation system built using Python. The system preprocesses movie data to generate meaningful tags for each movie and can recommend movies based on similarity.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Features](#features)
- [Usage](#usage)
- [Libraries Used](#libraries-used)
- [How It Works](#how-it-works)

---

## Dataset
The project uses the following datasets:
1. `tmdb_5000_movies.csv`: Contains information about movies, including genres, keywords, overview, etc.
2. `tmdb_5000_credits.csv`: Contains additional information, such as cast and crew.

These datasets are merged on the `title` column for preprocessing.

---

## Preprocessing
The following preprocessing steps are applied to the data:
1. **Merging DataFrames**: Combined `movies` and `credits` datasets on the `title` column.
2. **Selecting Relevant Columns**:
   - `movie_id`
   - `title`
   - `overview`
   - `genres`
   - `keywords`
   - `cast`
   - `crew`
3. **Handling Missing Values**: Removed rows with missing data.
4. **Transforming Data**:
   - Converted JSON-like strings in `genres`, `keywords`, `cast`, and `crew` into lists of relevant names.
   - Limited `cast` to the top 3 actors.
   - Extracted the director from the `crew` column.
   - Tokenized `overview` into words.
   - Removed spaces from names to standardize tags.
5. **Combining Features**: Merged `overview`, `genres`, `keywords`, `cast`, and `crew` into a single `tags` column.
6. **Stemming**: Applied stemming to reduce words to their root forms (e.g., "running" -> "run").

The final preprocessed data is saved as `export2.csv`.

---

## Features
- Preprocessing pipeline for movie data.
- Generates a single `tags` column combining multiple attributes of a movie.
- Tags are lowercased and stemmed for consistency.
- Ready-to-use dataset for building a recommendation model.

---

## Usage
1. Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project directory.
2. Run the script to preprocess the data:
   ```bash
   python preprocess_movies.py
