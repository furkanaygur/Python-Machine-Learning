# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

datas = pd.read_csv("datas.csv")


le = preprocessing.LabelEncoder()
ohe = preprocessing.OneHotEncoder()

# Age
Age = datas.iloc[:,1:4].values

# Gender
gender = datas.iloc[:,-1:].values
gender[:,-1] = le.fit_transform(datas.iloc[:,-1])
gender = ohe.fit_transform(gender).toarray()

print(gender)

# Country
country = datas.iloc[:,0:1].values
country[:,0] = le.fit_transform(datas.iloc[:,0])
country = ohe.fit_transform(country).toarray()

print(country)

# Result
result = pd.DataFrame(data=Age, index=range(22) , columns= ['height','weight','age'] )
print(result)

result2 = pd.DataFrame(data=country, index=range(22), columns= ['tr','fr','us'] )
print(result2)

result3 = pd.DataFrame(data= gender[:,:1], index=range(22), columns=['Gender',])
print(result3)

s = pd.concat([result2,result], axis=1)
s2 = pd.concat([s,result3], axis=1)


x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size = 0.33, random_state = 0 )


regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

height= s2.iloc[:,3:4].values
print(height)


left = s2.iloc[:,:3]
right = s2.iloc[:,4:]

data = pd.concat([left,right] , axis=1 )



x_train, x_test, y_train, y_test = train_test_split(data,height, test_size = 0.33, random_state = 0 )

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_pred = regressor2.predict(x_test)