#python 3
#multivariables regression using statsmodel api
#make sure we eliminate the dependency between the parameters
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#using ordinary least squares lib
import statsmodels.api as sm

df = pd.read_excel("cars.xls")

#ignore the categorical data, using both ordinals and numerical

#x is a list of features, y is the target
x = df[["Mileage", "Cylinder", "Liter"]]
y = df["Price"]

#normalization using sklearn
sc = StandardScaler()
#x[["Mileage", "Cylinder", "Liter"]] = sc.fit_transform(x[["Mileage", "Cylinder", "Liter"]].as_matrix)
x = sc.fit_transform(x)
hypothesis = sm.OLS(y,x).fit()
print(hypothesis.summary())


#seeing the affect of a single parameter
#grouping price (mean) based on similar liter sizes
param = "Liter"
print(y.groupby(df[param]).mean())

plt.scatter(df[param], df["Price"])

