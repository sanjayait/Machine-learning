# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:27:24 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd

# Read data file
df=pd.read_csv("spam_ham.csv")
col=list(df.columns)

df.drop(["Unnamed: 0","label_num"],axis=1,inplace=True)

# Delete duplicate values
df.drop_duplicates(inplace=True)

df.groupby("label").describe()

# Convert text to number in lable
df["label_n"]=df["label"].apply(lambda x:1 if x=="spam" else 0)

# Split dataset into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(df["text"],df["label_n"],test_size=0.20)


# Feature extraction text --CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
xtrain_count=cv.fit_transform(xtrain.values)
xtrain_count.toarray()[:3]

# Create a model
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(xtrain_count,ytrain)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = cv.transform(emails)
model.predict(emails_count)

# Check accuracy
xtest_count=cv.transform(xtest)
model.score(xtest_count,ytest)

# Sklearn pipeline
from sklearn.pipeline import Pipeline
clf=Pipeline([("vectorizer",CountVectorizer()),("nb",MultinomialNB())])
clf.fit(xtrain,ytrain)

clf.score(xtest,ytest)
clf.predict(emails)
