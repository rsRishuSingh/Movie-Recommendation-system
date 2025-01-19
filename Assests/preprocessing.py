import numpy as np
import pandas as pd
from ast import literal_eval
from nltk.stem.porter import PorterStemmer



def convert(obj):
    list1 = []
    for i in literal_eval(obj):
        list1.append(i['name'])
    return list1
def convert3(obj):
    list2 = []
    j = 0;
    for i in literal_eval(obj):
        if( j < 3):
            list2.append(i['name'])
            j +=1
        else:
            break
    return list2

def getdirector(obj):
    list3 = []
    for i in literal_eval(obj):
        if i['job']=='Director':
            list3.append(i['name'])
            break
    return list3
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# print(credits.head(1), end='\v')
# print(credits.head(1)['cast'].values)
movies = movies.merge(credits, on='title')      # merging dataframes
# print(movies.head(1))
print('hello')
# print(movies.info())                # all columns
'''we have included following columns:
1.genres
2.id
3.keywords
4.title
5.overview
6.crew
7.cast'''

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast','crew']]
# print(movies)
# print(movies.isnull().sum)
movies.dropna(inplace=True)
# print(movies.isnull().sum)

print(movies.duplicated().sum())

# print(movies['genres'].apply(convert))
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

movies['cast'] = movies['cast'].apply(convert3)
# print(movies['cast'])
movies['crew'] = movies['crew'].apply(getdirector)
# print(movies['crew'])


# print(movies['overview'][3])
movies['overview'] = movies['overview'].apply(lambda x : x.split())
# print(movies['overview'][3])

movies['genres'] = movies['genres'].apply(lambda x : [i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x : [i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x : [i.replace(' ','') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x : [i.replace(' ','') for i in x])

# print(movies['crew'])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
# print(movies['tags'])

new_df = movies[['movie_id', 'title', 'tags']]
# print(new_df.head(2))
new_df['tags'] = new_df['tags'].apply(lambda x : " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x : x.lower())

print(new_df.head(20))



#steming    
ps = PorterStemmer()
def stem(text):
    y =[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)




new_df['tags'] = new_df['tags'].apply(stem)

# new_df.to_csv('export2.csv', index= False)


# preprocessing done