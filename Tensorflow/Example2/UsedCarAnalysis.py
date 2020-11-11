# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error


datas = pd.read_excel('merc.xlsx')

# ******************** data slicing

datas = datas.drop('transmission', axis=1)

print(datas.describe())
print(datas.isnull().sum())

sbn.distplot(datas['price'])
plt.show()
sbn.countplot(datas['year'])
plt.show()
print(datas.corr()['price'].sort_values())

print(datas.sort_values('price', ascending=False).head(20))

print(len(datas) * 0.01)

datas = datas.sort_values('price', ascending=False).iloc[131:]
print(datas.describe())

sbn.distplot(datas['price'])
plt.show()

print(datas.groupby('year').mean()['price'])

datas = datas[datas['year'] != 1970]
print(datas.groupby('year').mean()['price'])


# ************************ train

x = datas.drop('price', axis=1).values
y = datas['price'].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = Sequential()

model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


model.fit(x= x_train, y = y_train , validation_data=(x_test, y_test), batch_size=250, epochs=300)

# ******************* Results 

loss_datas = pd.DataFrame(model.history.history)
print(loss_datas)

loss_datas.plot()
plt.show()

predict_list = model.predict(x_test)

print(mean_absolute_error(y_test, predict_list))

plt.scatter(y_test, predict_list)
plt.plot(y_test,y_test,'y--')
plt.show()


# Check Result with an example

print(datas.iloc[2])

example_test_data = datas.drop('price', axis=1).iloc[2]

example_test_data = scaler.transform(example_test_data.values.reshape(-1,5))

print(model.predict(example_test_data))