# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:49:50 2021

@author: Sanjay
"""

# Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np

# Load dataset
from sklearn.datasets import load_digits
digits=load_digits()

# Split train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(digits.data,digits.target,test_size=0.15)

# Use LogisticRegression classifier
lr=LogisticRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)

# Use RandomForest classifier
rfc=RandomForestClassifier()
rfc.fit(xtrain,ytrain)
rfc.score(xtest,ytest)

# Use Supoort vector classifier(SVC)
svmc=SVC()
svmc.fit(xtrain,ytrain)
svmc.score(xtest,ytest)

# Define a function
def get_score(model,xtrain,xtest,ytrain,ytest):
    model.fit(xtrain,ytrain)
    return model.score(xtest,ytest)

get_score(LogisticRegression(),xtrain,xtest,ytrain,ytest)
get_score(SVC(),xtrain,xtest,ytrain,ytest)
get_score(RandomForestClassifier(),xtrain,xtest,ytrain,ytest)

# StratifiedkFold
from sklearn.model_selection import StratifiedKFold
folds=StratifiedKFold()

score_lr=[]
score_svm=[]
score_rf=[]

for i,j in folds.split(digits.data,digits.target):
    X_train,X_test,y_train,y_test=digits.data[i],digits.data[j],digits.target[i],digits.target[j]
    score_lr.append(get_score(LogisticRegression(),X_train,X_test,y_train,y_test))
    score_svm.append(get_score(SVC(),X_train,X_test,y_train,y_test))
    score_rf.append(get_score(RandomForestClassifier(),X_train,X_test,y_train,y_test))

# K-Folds Cross-validator
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(),digits.data,digits.target)
cross_val_score(SVC(),digits.data,digits.target)
cross_val_score(RandomForestClassifier(),digits.data,digits.target)
