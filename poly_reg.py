#python 3
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sklearn
np.random.seed(2)
page_speed = np.random.normal(3,1,1000)
purchase_amount = np.random.normal(50,10,1000)/page_speed
# Y = W/X Where both W and X are RV
plt.scatter(page_speed,purchase_amount)
plt.close()

#using the polyfit class
#a very useful class, need to understand all its methods and attributes
x = np.array(page_speed)
y = np.array(purchase_amount)
p4 = np.poly1d(np.polyfit(x,y,4))   #4th degrees poly modelling    

plt.scatter(x,y)
#x_p4 = [i for i in range(8)]
x_p4 = np.linspace(0,8,1000)
plt.scatter(x_p4,p4(x_p4),c='r')    #evaluating poly1d at x points
plt.show()                          #result is clearly overfitting with p=4 ;)

rsquared = sklearn.r2_score(y, p4(x))
print("r2 = ", rsquared)