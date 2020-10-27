# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datas = pd.read_csv('datas.csv')

print(datas)

# # Fill the empty space with an average
# # First Way
# df = pd.DataFrame(datas)

# def avg(df):
#     total = df['age'].sum()
#     piece = df['age'].size - df['age'].isnull().sum()
#     return total / piece

# result = df.fillna(value=avg(df))
# print(result) 

# # Second Way

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
Age = datas.iloc[:,1:4].values
print("Result = ",Age)
imputer = imputer.fit(Age[:,1:4])
Age[:,1:4] = imputer.transform(Age[:,1:4])
print("Result = ",Age)