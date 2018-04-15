#python3

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from pylab import *

#premade dataset from sklearn with 3 kinds of flowers and 4 dims
iris = load_iris()

#print(iris.data.shape)
#n = sample size
n, features_size = iris.data.shape

X = iris.data
#always normalize data with whiten = True
pca = PCA(n_components=2, whiten=True)
pca = pca.fit(X)
new_X = pca.transform(X)
print(iris.target_names) #items needed to be classified
#print(new_X.explained_variance_)

#adding up the components to see the ratio in which we have preserved our data
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_)) #better when closer to 1
#print(new_X.n_components_)


#a = zip(range(len(iris.target_names)),'rgb',iris.target_names)



colors = cycle('rgb')
target_ids = range(len(iris.target_names))
pl.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
    print(i,c,label)
    #plotting our 2 components into (x,y) with the color of targets
    #with boolean masking of targets for filtering
    pl.scatter(new_X[iris.target == i, 0], new_X[iris.target == i, 1],
        c=c, label=label)    
    #original
    #pl.scatter(X[iris.target == i, 0], X[iris.target == i, 1],
        #c=c, label=label)
pl.legend()
pl.show()
