#python 3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#data fabrication with linear relationship
page_speed = np.random.normal(3,3,1000)
purchase_amount = 100 - (page_speed + np.random.normal(0, 1, 1000))*3
#modelled as X = 100 - (Y + W)*3 where Y and W are RV, with W noise ~N(0,0.1)
#if sigma of W is too high, correlation coefficient won't be as clean as they can be
plt.scatter(page_speed,purchase_amount)
plt.show()
plt.close()
slope, intercept, r_val, p_val, std_err = stats.linregress(page_speed,purchase_amount)
rsquared = r_val**2

#hypothesis based on the extracted parameters
def predict(x, slope, intercept):
    return slope*x + intercept

h = predict(page_speed,slope,intercept)

plt.scatter(page_speed, purchase_amount)
plt.plot(page_speed,h,c='r')
plt.show()