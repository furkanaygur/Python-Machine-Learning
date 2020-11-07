# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

datas = pd.read_csv("customers.csv")

x = datas.iloc[:,3:].values

kmenas = KMeans(n_clusters=3, init='k-means++')
kmenas.fit(x)

print(kmenas.cluster_centers_)

results = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123 )
    kmeans.fit(x)
    results.append(kmeans.inertia_)
    
plt.plot(range(1,11), results)