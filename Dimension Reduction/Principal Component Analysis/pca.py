# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Datas
datas = pd.read_csv('Wine.csv')

x = datas.iloc[:,:13].values
y = datas.iloc[:,13:].values

# train, test datas
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# PCA
pca = PCA(n_components= 2)

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)

# Before PCA values
classifier = LogisticRegression(random_state= 0)
classifier.fit(X_train, y_train)

# After PCA values
classifier2 = LogisticRegression(random_state= 0)
classifier2.fit(X_train2, y_train)

# # Predict
# Before PCA 
y_pred = classifier.predict(X_test)

# After PCA
y_pred2 = classifier2.predict(X_test2)


# Result

# Actual / Before PCA
cm = confusion_matrix(y_test, y_pred)
print('Before PCA confusion matrix: \n',cm)

# Actual / After PCA
cm2 = confusion_matrix(y_test, y_pred2)
print('\n After PCA confusion matrix: \n',cm2)

# Before PCA / After PCA
cm3 = confusion_matrix(y_pred, y_pred2)
print('\n Before PCA / After PCA confusion matrix: \n', cm3)

