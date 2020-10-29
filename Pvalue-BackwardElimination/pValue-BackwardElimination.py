# -*- coding: utf-8 -*-
"""

@author: furkan
"""


import pandas as pd
from sklearn import preprocessing
import numpy as np 
import statsmodels.api as sm

datas = pd.read_csv("datas.csv")


le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()



Age = datas.iloc[:,1:4].values
result = pd.DataFrame(data=Age, index=range(22) , columns= ['height','weight','age'] )

country = datas.iloc[:,0:1].values
country[:,0] = le.fit_transform(datas.iloc[:,0])
country = ohe.fit_transform(country).toarray()
result2 = pd.DataFrame(data=country, index=range(22), columns= ['tr','fr','us'] )

gender = datas.iloc[:,-1:].values
gender[:,-1] = le.fit_transform(datas.iloc[:,-1])
gender = ohe.fit_transform(gender).toarray()
result3 = pd.DataFrame(data= gender[:,:1], index=range(22), columns=['Gender',])

s = pd.concat([result2,result], axis=1)
s2 = pd.concat([s,result3], axis=1)

left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

data = pd.concat([left,right] , axis=1 )

x = np.append(arr = np.ones((22,1)).astype(int), values=data, axis=1 )

height= s2.iloc[:,3:4].values

x_list = data.iloc[:,[0,1,2,3,4,5]].values
x_list = np.array(x_list, dtype=float)
model = sm.OLS(height, x_list).fit()
print(model.summary())
# x6 p value = 0.717 and we delete this 


x_list = data.iloc[:,[0,1,2,3,5]].values
x_list = np.array(x_list, dtype=float)
model = sm.OLS(height, x_list).fit()
print(model.summary())
# x5 p value = 0.031 actualy its ok but i delete this 

x_list = data.iloc[:,[0,1,2,3]].values
x_list = np.array(x_list, dtype=float)
model = sm.OLS(height, x_list).fit()
print(model.summary())