# -*- coding: utf-8 -*-
"""


@author: furkan
"""

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


datas = pd.read_csv('customers.csv')

x = datas.iloc[:,3:].values

ac = AgglomerativeClustering(n_clusters=4, linkage='ward', affinity='euclidean' )
y_pred = ac.fit_predict(x)

print(y_pred)


plt.scatter(x[y_pred==0,0], x[y_pred == 0,1], s=100, c='red')
plt.scatter(x[y_pred==1,0], x[y_pred == 1,1], s=100, c='green')
plt.scatter(x[y_pred==2,0], x[y_pred == 2,1], s=100, c='blue')
plt.scatter(x[y_pred == 3,0], x[y_pred == 3,1],s=100, c='black')
plt.show()


##
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()