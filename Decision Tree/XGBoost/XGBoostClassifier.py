# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


# datas
datas = pd.read_csv('Churn_Modelling.csv')

x = datas.iloc[:,3:13].values
y = datas.iloc[:,13:].values

# Encoding
le = LabelEncoder()

x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])


ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

# train, test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# XGBoost
xgb= XGBClassifier()
xgb.fit(x_train, y_train)

y_pred = xgb.predict(x_test)

# Result
cm  = confusion_matrix(y_test, y_pred)
print(cm)