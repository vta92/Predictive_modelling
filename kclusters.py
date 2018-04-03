# python 3
#need to revisit and plot the centroids
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import sklearn.preprocessing






#n is the size of our data set, with k clusters
def createData(n,k):
    np.random.seed(2)
    cluser_avg_size = n/k #size of the clusters on average
    result = [] #list of list with the inner list being [income,age]

    for i in range(k):
        #draw a random sample b/t 100k-300k income, and 20-70 years old
        income_cen = np.random.uniform(100000.0,300000.0)
        age_cen = np.random.uniform(20.0,70.0)
        
        #append [income,age] with a given variation
        for j in range(int(cluser_avg_size)):
            result.append([np.random.normal(income_cen, 30000.0),np.random.normal(age_cen, 4.0)])
            #print(result)
    result = np.array(result)
    return result

if __name__ == "__main__":
    data = createData(1000,4)
    #plt.scatter(data.transpose()[1], data.transpose()[0])
    model = sklearn.cluster.KMeans(n_clusters = 3)   #init 3 centroids
    model = model.fit(sklearn.preprocessing.scale(data)) #data scaling due to differences between age/inc
    print(model.labels_) #looking at the clusters each data assigned to
    print(model.cluster_centers_)   #looking at the coordinates of the centroids
    

    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)

    plt.scatter(data[:,0], data[:,1], c= model.labels_.astype(float))
    plt.scatter(x_cen,y_cen)
#    plt.scatter()
    plt.show()