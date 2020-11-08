# -*- coding: utf-8 -*-
"""

@author: furkan
"""
# Breadth First Search

import pandas as pd
from libraries.apyori import apriori

datas = pd.read_csv('cart.csv',header=None)

x = []

for i in range(0,7501):
    x.append( [str(datas.values[i,j]) for j in range(0,19)] )

rules = apriori(x,min_support=0.01, min_confidence=0.2, min_lift=3, min_lenght=2 )

print(list(rules))
