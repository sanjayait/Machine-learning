# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 23:21:40 2021

@author: Sanjay
"""

import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits=load_digits()

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    
# Split train test
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(digits.data,digits.target,test_size=0.15,random_state=15)

# Create a model
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200) # default=100
rfc.fit(xtrain,xtest)

rfc.score(ytrain,ytest)

# Confusion matrix
y_predicted=rfc.predict(ytrain)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,y_predicted)
