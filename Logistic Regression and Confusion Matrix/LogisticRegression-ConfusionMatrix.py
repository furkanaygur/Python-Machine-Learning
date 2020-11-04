# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

datas = pd.read_csv("datas.csv")

x = datas.iloc[6:,1:4].values
y = datas.iloc[6:,4:].values


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0 )

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) # NOT fit_transform !


log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(y_pred)
print(y_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)