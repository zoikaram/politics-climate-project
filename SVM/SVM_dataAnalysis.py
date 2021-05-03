# -*- coding: utf-8 -*-
"""
Created on Sun May  2 20:14:56 2021

@author: dries
"""
from joblib import load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

print("start run:")
startTime = datetime.now().time()
print(startTime)

svm = load("svm")

data = pd.read_csv("./cleanUNData.csv")


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(data['text'].values.astype('U'))
Test_X_Tfidf = Tfidf_vect.transform(data['text'].values.astype('U'))

# predict the labels on validation dataset
data['climateLabel'] = svm.predict(Test_X_Tfidf)

groupedData = data[['folder', 'climateLabel']].groupby(['folder']).mean()

import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0, len(groupedData['climateLabel'].values), 1)
s = groupedData['climateLabel'].values
fig, ax = plt.subplots()
ax.plot(t, s, 'ro')

ax.set(xlabel='time (s)', ylabel='percent of sentences about climate',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()


print("end run:")
endTime = startTime = datetime.now().time()
print(endTime)
