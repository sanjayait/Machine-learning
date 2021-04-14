# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:06:00 2021

@author: Sanjay
"""
# Import libraries
import pandas as pd
from matplotlib import pyplot as plt

# Read file
df=pd.read_csv("HR_analytics.csv")
col=list(df.columns)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

# Data exploration and Visualization
left=df[df.left==1]
retained=df[df.left==0]

# Average numbers for all columns
df.groupby("left").mean()

corelation=df.corr()

#From above table we can draw following conclusions,

# 1--**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
# 2--**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
# 3--**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm
# 4-- **Some how Work_acciedent also get affected


# Impact of salary on employee retaintion
pd.crosstab(df.salary,df.left).plot(kind="bar")
# Above bar chart shows employees with high salaries are likely to  not leave the company

# Impact of department on employee retaintion
pd.crosstab(df.Department,df.left).plot(kind="bar")
# From above chart there seem to be some impact of department on employee retention but it is not major hence we will ignore department in our analysis


# From the data analysis so far we can conclude that we will use following variables as independant variables in our model
# 1--**Satisfaction Level**
# 2--**Average Monthly Hours**
# 3--**Promotion Last 5 Years**
# 4--**Salary**
# 5--**Work accident**

# Remove duplicates data
clean=df.drop_duplicates()

subdf=clean.drop(["last_evaluation","number_project","time_spend_company","left","Department"],axis="columns")


# Tackle salary dummy variable
salary_dummy=pd.get_dummies(subdf.salary,prefix="salary")
final=pd.concat([subdf,salary_dummy],axis='columns')
final.drop("salary",axis='columns',inplace=True)

# Split data into train and test
X=final
y=clean.left

from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(X,y,test_size=0.2,random_state=10)

# Create a Model
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(xtrain,xtest)

# Prediction
lor.predict(ytrain)

# Check accuracy of model
lor.score(ytrain,ytest)
