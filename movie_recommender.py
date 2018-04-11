#python3
#item based recommender
import pandas as pd
import numpy as np


col = ['user_id','movie_id','ratings']
col2 = ['movie_id','title']
url = 'ml-100k/u.data'
url2 = 'ml-100k/u.item'

ratings = pd.read_csv(url, sep='\t', names=col, usecols=range(3), encoding="ISO-8859-1")
movies = pd.read_csv(url2, sep='|', names=col2, usecols=range(2), encoding="ISO-8859-1")

#the difference between merge (ratings, movies) and (movies,rating) is?
#nothing but representations
db = pd.merge(movies,ratings)
print(db.isnull().sum())

#creating a sparse matrix with cols = title and rows= user_id
db = pd.pivot_table(db, values='ratings', index='user_id',columns='title')

#look at diagonals and cor(a,b) = cor(b,a). It works
#correlations = db.corr()
#making sure that each 2 movies pairs are rated by at least 50+ users
correlations = db.corr(method='pearson', min_periods=50)

#############################################################################
#individual users
#first user rated 3 movies only
#print(db.iloc[0].dropna())
current_user = db.iloc[0].dropna()

similar_movies = pd.Series()

for i in range(0,len(current_user.index)):#go through all titles of curr existing rating
    print(current_user.index[i])    #print existing rating
    similar = correlations[current_user.index[i]].dropna()
    #the line above basically query the current existing movie title column
    #and see any resulting movies recommendations we have in that sparse matrix
    
    #scaling similar candidates, if i like my current movie 5*, 
    #then i scale the most similar stuff with 5x
    #if current is 1*, then no scaling on correlated movies to that bad title
    similar2 = similar.map(lambda x: x*current_user[i])
    similar_movies = similar_movies.append(similar2)


#similar_movies.duplicated().sum() shows repeated values
similar_movies.groupby(similar_movies.index).mean()
similar_movies = similar_movies.sort_values(ascending=False)
print(similar_movies.head(10))



