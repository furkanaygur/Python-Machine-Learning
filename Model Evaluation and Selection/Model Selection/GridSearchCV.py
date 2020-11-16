# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# datas
datas = pd.read_csv('Social_Network_Ads.csv')

x = datas.iloc[:,2:4].values
y = datas.iloc[:,4:].values

#  train, test
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=0 )

# Scaling
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

svc = SVC(kernel = 'rbf', random_state= 0 )

svc.fit(X_train, y_train)


# GridSearchCV

params = [{'C':[1,2,3,4,5], 'kernel':['linear']}, 
          {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]}
          ] 

gridsearch = GridSearchCV(estimator=svc, 
                          param_grid=params,
                          scoring='accuracy',
                          cv=10,
                          n_jobs = -1 )

Grid_Search_Results = gridsearch.fit(X_train, y_train)

print('Best Score: ' ,Grid_Search_Results.best_score_)
print('Best Params: ', Grid_Search_Results.best_params_)

'''
 Result ==>     Best Score:  0.9142857142857143
                Best Params:  {'C': 1, 'gamma': 1, 'kernel': 'rbf'}
'''