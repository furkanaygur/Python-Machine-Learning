# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle 

datas = pd.read_csv('sales.csv')

x = datas.iloc[:,0:1].values
y = datas.iloc[:,1:].values
 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

lr = LinearRegression()
lr.fit(x_train, y_train)

print(lr.predict(x_test))


# # Saving Model
folder = 'model1'

pickle.dump(lr,open(folder,'wb'))


# # Running with the saved model
model = pickle.load(open(folder,'rb'))

print('result from saved model: \n', model.predict(x_test))
