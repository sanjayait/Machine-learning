# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 18:09:37 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt


# Read data file
df=pd.read_csv("titanic.csv")
col=list(df.columns)
df.columns

live=df[df.Survived==1]
die=df[df.Survived==0]

# Exploratory data analysis
df.info()
df.drop_duplicates(inplace=True)
df.describe()
corelation=df.corr()

df.drop(columns=['Name_wiki',
       'Age_wiki', 'Hometown', 'Boarded', 'Destination', 'Lifeboat', 'Body',
       'Class','PassengerId','WikiId','Name',],axis=1,inplace=True)

df.groupby("Survived").mean()
num_df=df.select_dtypes(exclude="object")
cat_df=df.select_dtypes(include="object")
cat_df.drop(["Ticket","Cabin","Embarked"],axis=1,inplace=True)

pd.crosstab(index=df.Sex,columns=df.Survived,dropna=True).plot(kind="bar")

# Preprocessing
subdf=pd.concat([num_df,cat_df],axis=1)
droped=subdf.dropna(subset=["Survived"])
droped.info()
droped.Age.mean()
droped.Age.median()
droped.Age.fillna(droped.Age.median(),inplace=True)

# Tackel with categoraical variable
droped.Sex=droped.Sex.map({'male':1,'female':2})
x = droped.drop(["Survived"],axis=1)
y=droped.Survived

x.Age.fillna(x.Age.median(),inplace=True)
# Split train and test
from sklearn.model_selection import train_test_split
x1,y1,x2,y2=train_test_split(x,y,test_size=.15,random_state=10)

# Create a Model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# From logistic regression
lor=LogisticRegression()
lor.fit(x1,x2)

lor.score(y1,y2)

# From decission tree
dt=DecisionTreeClassifier()
dt.fit(x1,x2)

dt.score(y1,y2)
