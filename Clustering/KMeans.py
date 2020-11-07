# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

datas = pd.read_csv("customers.csv")

x = datas.iloc[:,3:].values

kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)

results = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123 )
    kmeans.fit(x)
    results.append(kmeans.inertia_)
    
plt.plot(range(1,11), results)
plt.show()



kmeans2 = KMeans(n_clusters=4, init='k-means++', random_state=123 )
y_pred = kmeans2.fit_predict(x)

print(y_pred)

plt.scatter(x[y_pred == 0,0], x[y_pred == 0,1],s=100, c='red')
plt.scatter(x[y_pred == 1,0], x[y_pred == 1,1],s=100, c='green')
plt.scatter(x[y_pred == 2,0], x[y_pred == 2,1],s=100, c='blue')
plt.scatter(x[y_pred == 3,0], x[y_pred == 3,1],s=100, c='black')