# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sbn
from sklearn.metrics import mean_absolute_error, mean_squared_error


datas = pd.read_excel('BicyclesPrices.xlsx')

x = datas.iloc[:,1:].values
y = datas.iloc[:,:1].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33,random_state=15)

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

model = Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')

model.fit(x_train_scaled, y_train, epochs=300)

loss = model.history.history['loss']

sbn.lineplot(x=range(len(loss)), y=loss)

print("Train loss = ",model.evaluate(x_train_scaled, y_train, verbose=0))
print("Test loss = ",model.evaluate(x_test_scaled, y_test, verbose=0))


testPredicts = model.predict(x_test_scaled)
print(testPredicts)

DF_pred = pd.DataFrame(y_test, columns=['Real Price'])
testPredicts = pd.Series(testPredicts.reshape(330,))

DF_pred = pd.concat([DF_pred,testPredicts],axis=1)

DF_pred.columns= ['Price Real', 'Price Predict']

sbn.scatterplot( x='Price Real', y = 'Price Predict' , data = DF_pred)


print("Mean Absolute Error: ",mean_absolute_error(DF_pred['Price Real'], DF_pred['Price Predict']))
print("Mean Squared Error: ",mean_squared_error(DF_pred['Price Real'], DF_pred['Price Predict']))








