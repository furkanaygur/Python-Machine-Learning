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