# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

datas = pd.read_csv("salaries.csv")

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values


# Random Forest Regression
rf_reg = RandomForestRegressor(n_estimators=10, random_state=0 )
rf_reg.fit(x, y.ravel())

print("Random Forest R2 value: ",r2_score(y, rf_reg.predict(x)))

# Decision Tree Regression
dt_reg = DecisionTreeRegressor(random_state=0 )
dt_reg.fit(x, y)

print("Decision Tree R2 Value: ", r2_score(y, dt_reg.predict(x))) 

# Support Vector Regression
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
y_scaled = np.ravel(sc.fit_transform(y.reshape(-1,1)))

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)


print("SVR R2 Value: ", r2_score(y_scaled, svr_reg.predict(x_scaled)))


# Polynomial Regression R2 Value
pr = PolynomialFeatures(degree=4)
x_poly = pr.fit_transform(x)

lr = LinearRegression()
lr.fit(x_poly,y)

print("Polynomial Regression R2 Value: ", r2_score(y, lr.predict(x_poly)))

# Linear Regression R2 Value
lr2 = LinearRegression()
lr2.fit(x,y)

print("Linear Regression R2 Value: ", r2_score(y,lr2.predict(x)))
