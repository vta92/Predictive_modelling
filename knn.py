#python3
import numpy as np
import pandas as pd
import scipy.spatial
import operator

cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=cols, usecols=range(3))
#print(ratings.head())

#summing to see the characteristics with size, and average
movie_knn = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
#print(movie_knn.head())

#take only the aggregated size column and normalize it
movie_knn_size = pd.DataFrame(movie_knn['rating']['size'])
max = movie_knn_size['size'].max()
min = movie_knn_size['size'].min()
movie_knn_size_norm = (movie_knn_size['size'] - min)/(max - min)
movie_knn_size_norm = pd.DataFrame(movie_knn_size_norm)

movie_dict = {}

with open('ml-100k/u.item', encoding = "ISO-8859-1") as f:
    temp = ''
    for line in f:
        #print(line) #look at the output of lines
        fields = line.rstrip('\n').split('|') #sep = '|'
        movie_id =  int(fields[0])
        name = fields[1]
        genres = fields[5:]
        genres = [i for i in genres]
        #put genres into an array of binary
        #key = movie id number
        #values = tuples(movie name, genre binary list, normalized size, mean rating)
        movie_dict [movie_id] = (name, np.array(list(genres)), 
                   movie_knn_size_norm.loc[movie_id].get('size'), 
                   movie_knn.loc[movie_id].rating.get('mean'))
        #print(name, movie_knn_size_norm.loc[movie_id].get('size'), movie_knn.loc[movie_id].rating.get('mean'))

#defining the distance between genre + distance between the views/seen size
#dict with key input: ie movie_dict[1682]
#v1, v2 distance = v2,v1 distance
def Distance(movie1,movie2):
    #genre distance
    genre1 = list(map(float, movie1[1]))
    #print(genre1)
    genre2 = list(map(float, movie2[1]))
    #cosine function from scipy.distance needs to be fed with non-np related array
    genre_dist = scipy.spatial.distance.cosine(genre1, genre2)
    popularity_dist = abs(movie1[2] - movie2[2]) #popularity/seen size
    quality_dist = abs(movie1[3] - movie2[3])/5 #normalization
    #not a recommender system, so we want to penalize for quality difference
    result = abs((genre_dist)*3 + (popularity_dist) - quality_dist*1.5)
    return result
   
#test = Distance(movie_dict[2], movie_dict[4])


def k_neighbors(movie_id, K):

    distances = []
    for i in movie_dict:
        if (i != movie_id): #not itself, with every other movie
            dist = Distance(movie_dict[movie_id], movie_dict[i])
            distances.append([i,dist]) #tuples
            #print(dist)
    distances.sort(key=lambda x: x[1])   #sort second elements of the distances list
    
    #choosing the nearest K neighbors sorted distant results
    neighbors = []
    #append the id of the movies with nearest top n distances
    for n in range(K):
        neighbors.append(distances[n][0])
    return neighbors


#test = k_neighbors(1682,10)
#movie_dict =>(movie name, genre binary list, normalized size, mean rating)
def get_knn(movie_id, k):
    #Jean de Florette :) 165. Trying to learn French
    knn = k_neighbors(movie_id, k)
    mean_rating = 0
    for i in knn:
        mean_rating += movie_dict[i][3] #rating
        print (movie_dict[i][0], str(movie_dict[i][3]))
    return mean_rating/k

######################################################################
#Jean de Florette = 165
k = 10
movie_id = 165
result = get_knn(movie_id,k)
print('Rating for current movie:\n',str(movie_dict[movie_id][0]) +
      " : " + str(movie_dict[movie_id][3]) )
print('Average rating for all those movies: '+ str(result))

