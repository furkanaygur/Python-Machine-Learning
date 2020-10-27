# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:54:23 2020

@author: furkan

"""

import pandas as pd
from sklearn import preprocessing

datas = pd.read_csv("datas.csv")

countries = datas.iloc[:,0:1].values
print(countries)

le = preprocessing.LabelEncoder()
countries[:,0] = le.fit_transform(datas.iloc[:,0])
print(countries)

ohe = preprocessing.OneHotEncoder()
countries = ohe.fit_transform(countries).toarray()
print(countries)


# # Data Mergeging

result = pd.DataFrame(data=countries, index=range(22), columns=['tr','fr','us'])
print(result)

Age = datas.iloc[:,1:4].values

result2= pd.DataFrame(data=Age, index=range(22), columns=['height','weight','age'])
print(result2)

gender = datas.iloc[:,-1].values
print(gender)

result3 = pd.DataFrame(data= gender, index=range(22), columns=['gender'])
print(result3)
                      
s =  pd.concat([result,result2], axis=1)
print(s)

s2 = pd.concat([s,result3], axis=1)
print(s2)