# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 22:33:36 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd
from sklearn.datasets import load_wine

wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)

# Split train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df,wine.target,test_size=0.15)

# Create a model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

mnb=MultinomialNB()
gnb=GaussianNB()
bnb=BernoulliNB()

mnb.fit(xtrain,ytrain)
gnb.fit(xtrain,ytrain)
bnb.fit(xtrain,ytrain)

mnb.score(xtest,ytest)
gnb.score(xtest,ytest)
bnb.score(xtest,ytest)

# K-Fold validater
from sklearn.model_selection import cross_val_score
cross_val_score(mnb,df,wine.target)
cross_val_score(gnb,df,wine.target)
cross_val_score(bnb,df,wine.target)

# Average
gm=cross_val_score(gnb,df,wine.target)*100
final_score=sum(gm)/5
