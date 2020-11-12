# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd 
import math
import matplotlib.pyplot as plt

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
prizes = [0] * d  # Ri(n)
clicks = [0] * d # Ni(n)
chosen = []
total = 0

for n in range(0, N):
    max_ucb = 0
    ad = 0
    for i in range(1, d):
        if clicks[i] > 0:     
            mean = prizes[i] / clicks[i]
            delta = math.sqrt(3/2 * math.log(i)/clicks[i])    
            ucb = mean + delta 
        else: 
            ucb = N*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad = i
    
    chosen.append(ad)
    clicks[ad] = clicks[ad] + 1
    prize = datas.values[n, ad]
    prizes[ad] = prizes[ad] + prize
    total = total + prize

print('Total Prize: ', total)


plt.hist(chosen)
plt.show()