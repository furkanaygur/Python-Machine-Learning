# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

datas = pd.read_csv("salaries.csv")

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values

sc = StandardScaler()
x_scaled = sc.fit_transform(x)
y_scaled = sc.fit_transform(y)


svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_scaled, y_scaled)

plt.scatter(x_scaled, y_scaled, color='black')
plt.plot(x_scaled, svr_reg.predict(x_scaled),   color='red')


print(svr_reg.predict(x_scaled))