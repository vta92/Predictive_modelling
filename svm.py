#python 3

import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.preprocessing
import sklearn.svm
import sklearn.datasets

#n is the size of our data set, with k clusters
#copied from the cluster data generation part
def createData(n,k):
    np.random.seed(10)
    cluser_avg_size = n/k #size of the clusters on average
    result = [] #list of list with the inner list being [income,age]
    indx = []

    for i in range(k):
        #draw a random sample b/t 100k-300k income, and 20-70 years old
        income_cen = np.random.uniform(100000.0,300000.0)
        age_cen = np.random.uniform(20.0,70.0)
        
        #append [income,age] with a given variation
        for j in range(int(cluser_avg_size)):
            result.append([np.random.normal(income_cen, 30000.0),np.random.normal(age_cen, 4.0)])
            #print(result)
            indx.append(i)
    result = np.array(result)
    indx = np.array(indx)
    return result, indx

#contour plot helper function
def plotPredictions(clf,X,Y):
    xx, yy = np.meshgrid(np.arange(0, 400000, 10),
                     np.arange(10, 100, 0.5))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    #plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.scatter(X[:,0], X[:,1], c=Y.astype(np.float))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.show()

if __name__ == "__main__":
    data,indx = createData(100,5)
    #degree is only useful for poly kernel
    #sigmoid is especially good for 2 types classification (nn)
    model = sklearn.svm.SVC(degree=3, kernel='linear', C=2)
    #model = sklearn.svm.SVC()
    model.fit(data,indx)
    
    
    plotPredictions(model, data, indx)

    
    predictions = [(200000,40), (300000,60),(25000,20),(80000,23)]    
    for i in predictions:
        print(model.predict([[i[0],i[1]]]))
