# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# data
datas = pd.read_csv('Wine.csv')

x = datas.iloc[:,:13].values
y = datas.iloc[:,13:].values

# train, test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)

# scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# LDA 
lda= LDA(n_components=2)

X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# After LDA
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda, y_train)


# # Predict
y_pred = classifier.predict(X_test)

# After LDA
y_pred_lda = classifier_lda.predict(X_test_lda)


# LDA and Original y_pred 
cm = confusion_matrix(y_pred, y_pred_lda)
print(' LDA confusion matrix: \n', cm)


 
 





