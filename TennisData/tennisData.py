# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 00:55:00 2020

@author: furka
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

datas = pd.read_csv("tennisData.csv")


datas2 = datas.apply(preprocessing.LabelEncoder().fit_transform)

outlooks = datas2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
outlooks = ohe.fit_transform(outlooks).toarray()

outlooksResult = pd.DataFrame(data=outlooks, index = range(14), columns=['overcast','rainy','sunny'])

lastDatas = pd.concat([outlooksResult, datas.iloc[:,1:3]], axis= 1 )
lastDatas = pd.concat([datas2.iloc[:,-2:] ,lastDatas ], axis=1)

x_train, x_test, y_train, y_test = train_test_split(lastDatas.iloc[:,:-1], lastDatas.iloc[:,-1:] , test_size=0.33, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


# Backward Elimination
x = np.append(arr = np.ones((14,1)).astype(int), values=lastDatas.iloc[:,:-1], axis=1 )

x_list = lastDatas.iloc[:,[0,1,2,3,4,5]].values
x_list = np.array(x_list, dtype=float )
model= sm.OLS(lastDatas.iloc[:,-1:].values, x_list).fit()
print(model.summary())


x_list = lastDatas.iloc[:,[1,2,3,4,5]].values
x_list = np.array(x_list, dtype=float )
model= sm.OLS(lastDatas.iloc[:,-1:].values, x_list).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)