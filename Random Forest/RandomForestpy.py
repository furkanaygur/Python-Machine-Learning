# -*- coding: utf-8 -*-
"""  

@author: furkan
"""

import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

datas = pd.read_csv("salaries.csv")

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values

rt_reg = RandomForestRegressor( n_estimators=10, random_state=0 )
rt_reg.fit(x, y.ravel())

plt.scatter(x, y, color='red')
plt.plot(x, rt_reg.predict(x), color='black' ) 
print(rt_reg.predict([[6.6]]))