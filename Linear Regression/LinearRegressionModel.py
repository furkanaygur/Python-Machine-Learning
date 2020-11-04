# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 20:01:31 2020

@author: furka
"""

import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

datas = pd.read_csv("sales.csv")

Months = datas[["months"]]
print(Months)

Sales = datas[["sales"]]
print(Sales)

x_train, x_test, y_train, y_test = train_test_split(Months, Sales, test_size=0.33, random_state=0 )

# sc = StandardScaler()

# X_train = sc.fit_transform(x_train)
# X_test = sc.fit_transform(x_test)

# Y_train = sc.fit_transform(y_train)
# Y_test = sc.fit_transform(y_test)

lr = LinearRegression()

lr.fit(x_train, y_train)

result =lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.title("Sales")
plt.xlabel("Months")
plt.ylabel("Sales")

plt.plot(x_test, lr.predict(x_test))




