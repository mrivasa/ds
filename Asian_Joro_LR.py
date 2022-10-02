from numpy import *
#import operator
import pandas as pd

observations = pd.read_csv('combined_joro_v2.csv')

Y = observations.iloc[:, -1]

X = observations.iloc[:, 0:19]


import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)


import matplotlib
import sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,max_iter=10000).fit(X_train, Y_train)
clfPredictLR = (clf.predict(X_test))

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']
confusion_matrix(Y_test, clfPredictLR)
print(classification_report(Y_test, clfPredictLR, target_names = target_names))