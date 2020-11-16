# -*- coding: utf-8 -*-
"""

@author: furkan
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

 

datas = pd.read_csv('RestaurantReviews.csv', error_bad_lines=False)
datas = datas.fillna(value=0)


# Preprocessing
nltk.download('stopwords')
ps = PorterStemmer()    

comments= []

for i in range(0, 716):
    comment = re.sub('[^a-zA-Z]',' ', datas['Review'][i])
    comment = comment.lower()
    comment = comment.split()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    comments.append(comment)

# Feature Extraction
cv = CountVectorizer(max_features=1000)

x = cv.fit_transform(comments).toarray()
y = datas.iloc[:,1].values

# train, test datas
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Machine Learning
gnb = GaussianNB()

gnb.fit(x_train, y_train)

# Predict
y_pred = gnb.predict(x_test)

# Result
cm = confusion_matrix(y_test, y_pred)
print(cm) # Accuracy = %68