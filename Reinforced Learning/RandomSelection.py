# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
import random

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
chosen = []
total = 0
for n in range(0, N):
    ad = random.randrange(d)
    chosen.append(ad)
    prize = datas.values[n, ad]
    total = total + prize