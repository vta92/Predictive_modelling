#python 3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



movies = pd.read_csv('ml-100k/u.item', sep='|', names = ['Movie ID', 'Movie Title'], 
                     usecols=range(2),encoding="ISO-8859-1")
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['User ID','Movie ID','Rating'], 
                      usecols = range(3), encoding="ISO-8859-1")
#intersection of the indices
db = pd.merge(movies,ratings, how='inner')

#sparse matrix (user_id x title, ratings as elements )
movies_ratings = db.pivot_table(index=['User ID'],
                                columns=['Movie Title'],values='Rating')
#print(movies_ratings.isnull().sum())


#correlations with all other movies from 1 specific movies
similar_movies = movies_ratings.corrwith(movies_ratings['Dumbo (1941)'])
similar_movies = similar_movies.dropna().sort_values(ascending = False)

#aggregating all ratings or similar movie tittle, take the mean of the ratings
movie_stats = db.groupby('Movie Title').agg({'Rating': [np.size, np.mean]})

popular_movies = movie_stats['Rating']['size'] >= 100   #watched by 100+ ppl
ranking_movies = movie_stats[popular_movies].sort_values([('Rating','mean')], ascending=False)

result = ranking_movies.join(pd.DataFrame(similar_movies, columns=['similarity'])).sort_values('similarity',ascending=False)
print(result.head(5))