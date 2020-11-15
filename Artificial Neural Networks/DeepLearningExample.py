# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

datas = pd.read_csv('Churn_Modelling.csv')

x = datas.iloc[:,3:13].values
y = datas.iloc[:,13:].values

# Encoding

le = LabelEncoder()

x[:,1] = le.fit_transform(x[:,1])

x[:,2] = le.fit_transform(x[:,2])

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])] , remainder="passthrough")

x = ohe.fit_transform(x)
x = x[:,1:]


# Train Test datas
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Artificial Neural Networks 
model = Sequential()

model.add(Dense(6, activation='relu', input_dim=11))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150)

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)

print(cm)








