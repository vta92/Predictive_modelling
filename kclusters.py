# python 3
import numpy as np
import matplotlib.pyplot as plt
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

data = createData(1000,3)

#plt.scatter(data.transpose()[1], data.transpose()[0])
