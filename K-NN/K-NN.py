# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

datas = pd.read_csv("datas.csv")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=5 , metric='minkowski') # Default values
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm  = confusion_matrix(y_test, y_pred)
print('Neighbors Count 5 Results:\n', cm)

knn2 = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn2.fit(X_train, y_train)

y_pred2 = knn2.predict(X_test)

cm = confusion_matrix(y_test, y_pred2)
print('Neighbors Count 1 Results:\n', cm)