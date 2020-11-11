# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

datas = pd.read_excel('maliciousornot.xlsx')

print(datas.info())

print(datas.describe())

print(datas.corr()['Type'].sort_values())


# ************************ train

x = datas.drop('Type', axis=1).values
y = datas['Type'].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(30,activation='relu'))
model.add(Dropout(0.6))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.6))

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.6))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss= 'binary_crossentropy',  optimizer='adam')


earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)

model.fit(x= x_train, y= y_train, validation_data=(x_test, y_test), verbose=1, callbacks=[earlyStopping] , epochs=700 )


# ******************* Results 

loss_datas = pd.DataFrame(model.history.history)
print(loss_datas)
loss_datas.plot()
plt.show()


y_pred = model.predict_classes(x_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))









