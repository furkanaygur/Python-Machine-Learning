# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


datas = pd.read_excel("iris.xls")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# Logistic Regression 
lr = LogisticRegression(random_state=0 )
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression Confusion Matrix: \n",cm)


# K-NN Classifier
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Knn Classifier Confusion Matrix: \n", cm)

# SVC (SVM Classifier)
svc = SVC(kernel='rbf' )
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("SVC Confusion Matrix: \n", cm)


# Desicion Tree Classifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Decision Tree Classifier Confusion Matrix: \n", cm)


# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Random Forest Classifier Confuison Matrix: \n", cm)








