# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
import random
import matplotlib.pyplot as plt

datas = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10 
ones = [0] * d
zeros = [0] * d
total = 0
chosen = []

for n in range(0, N):
    ad = 0
    max_thopson = 0
    
    for i in range(0, d):
        rand_beta = random.betavariate(ones[i] + 1 , zeros[i] + 1)
        
        if rand_beta > max_thopson:
            max_thopson = rand_beta
            ad = i
            
    chosen.append(ad)
    prize = datas.values[n, ad]
    if prize == 1:
        ones[ad] = ones[ad] + 1
    else:
        zeros[ad] = zeros[ad] +1
    
    total = total + prize
        
print('Total prize: ',total)

plt.hist(chosen)
plt.show