# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

datas = pd.read_csv("salaries.csv")

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values
 
#polynomial regression
lr = LinearRegression()

pr = PolynomialFeatures(degree= 2)

x_poly = pr.fit_transform(x)
lr.fit(x_poly,y)

plt.scatter(x,y, color='blue')
plt.plot(x, lr.predict(x_poly), color='red')
plt.show()

print("Degree:2 \n")
print("Degree 2 X_poly: \n",x_poly)

print("Education Level 11: ",lr.predict(pr.fit_transform([[11]])))
print("Education Level 6.6: ",lr.predict(pr.fit_transform([[6.6]])))

pr = PolynomialFeatures(degree= 4)

x_poly = pr.fit_transform(x)
lr.fit(x_poly,y)

plt.scatter(x,y, color='red')
plt.plot(x, lr.predict(x_poly), color='blue')
plt.show()

print("\nDegree:4  \n")
print("Degree 4 X_poly: \n",x_poly)

print("Education Level 11: ",lr.predict(pr.fit_transform([[11]])))
print("Education Level 6.6: ",lr.predict(pr.fit_transform([[6.6]])))
