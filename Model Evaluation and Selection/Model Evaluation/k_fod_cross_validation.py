# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# datas
datas = pd.read_csv('Social_Network_Ads.csv')

x = datas.iloc[:,2:4].values
y = datas.iloc[:,4:].values


# train, test datas
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# SVC
svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

# Result
cm = confusion_matrix(y_test, y_pred)
print('SVC confusion matrix: \n',cm)


success = cross_val_score(estimator=svc, X = X_train, y = y_train, cv=4)

print('Success: ',success.mean())

