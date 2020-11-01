# -*- coding: utf-8 -*-
"""


@author: furkan
"""

import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

datas = pd.read_csv("salaries.csv")

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values

dt = DecisionTreeRegressor(random_state=0)
dt.fit(x,y)

plt.scatter(x, y, color = "black")
plt.plot(x, dt.predict(x), color="red" )

print(dt.predict([[11]]))
print(dt.predict([[6.6]]))