# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

datas = pd.read_csv("newSalaries.csv")

x = datas.iloc[:,2:3].values
y = datas.iloc[:,5:].values
 
# Linear Regression
lr = LinearRegression()
lr.fit(x, y)

model = sm.OLS(lr.predict(x) ,x).fit()
print(" Linear OLS ".center(50,'*'))
print(model.summary())


# Polynomial Regression
pr = PolynomialFeatures(degree=4)
x_poly = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly, y)

model2 = sm.OLS( lr2.predict(x_poly), x).fit()
print(" Polynomial OLS ".center(50,'*'))
print(model2.summary())

# SVR 
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
y_scaled = sc.fit_transform(y)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)

model3 = sm.OLS(svr_reg.predict(x_scaled),x_scaled).fit()
print(" SVR OLS ".center(50,'*'))
print(model3.summary())


# Decision Tree
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x,y)

model4 = sm.OLS(dt.predict(x), x).fit()
print(' Decision Tree OLS '.center(50,'*'))
print(model4.summary())

# Random Forest 
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x,y)

model5 = sm.OLS(rf.predict(x),x).fit()
print(' Random Forest OLS '.center(50,'*'))
print(model5.summary())





