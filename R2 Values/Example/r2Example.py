# -*- coding: utf-8 -*-
"""

@author: furkan
"""

'''
3 paramteres results

linear R-squared: 0.903

Poşynomial R-squared: 0.680

SVR R-squared: 0.782

DT R-squared: 0.679

RF R-squared: 0.713


1 parameter results

linear R-squared: 0.942

Polynomial R-squared: 0.759

SVR R-squared: 0.770

DT R-squared: 0.751

RF R-squared: 0.719

'''


import pandas as pd 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


datas = pd.read_csv("newSalaries.csv")

print(datas.corr())

x = datas.iloc[:,2:5].values
y = datas.iloc[:,5:].values
 
# Linear Regression
lr = LinearRegression()
lr.fit(x, y)

model = sm.OLS(lr.predict(x) ,x).fit()
print(" Linear OLS ".center(50,'*'))
print(model.summary())

print(" \nLinear Regression R2 Value: ", r2_score(y,lr.predict(x)),'\n')

# Polynomial Regression
pr = PolynomialFeatures(degree=4)
x_poly = pr.fit_transform(x)

lr2 = LinearRegression()
lr2.fit(x_poly, y)

model2 = sm.OLS( lr2.predict(x_poly), x).fit()
print(" Polynomial OLS ".center(50,'*'))
print(model2.summary())

print("\nPolynomial Regression R2 Value: ", r2_score(y, lr2.predict(x_poly)),'\n')

# SVR 
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
y_scaled = sc.fit_transform(y)

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)

model3 = sm.OLS(svr_reg.predict(x_scaled),x_scaled).fit()
print(" SVR OLS ".center(50,'*'))
print(model3.summary())

print("\nSVR R2 Value: ", r2_score(y_scaled, svr_reg.predict(x_scaled)),'\n')


# Decision Tree
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x,y)

model4 = sm.OLS(dt.predict(x), x).fit()
print(' Decision Tree OLS '.center(50,'*'))
print(model4.summary())

print("\n Decision Tree R2 Value: ", r2_score(y, dt.predict(x)),'\n') 

# Random Forest 
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x,y)

model5 = sm.OLS(rf.predict(x),x).fit()
print(' Random Forest OLS '.center(50,'*'))
print(model5.summary())

print("\nRandom Forest R2 value: ",r2_score(y, rf.predict(x)),'\n')



