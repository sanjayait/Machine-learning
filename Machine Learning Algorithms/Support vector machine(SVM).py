# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:24:46 2021

@author: Sanjay
"""

import pandas as pd
from sklearn.datasets import load_iris

iris=load_iris()

# Create a dataframe
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"]=iris.target

# Add flower name column
df["flower_name"]=df["target"].apply(lambda x:iris.target_names[x])
col=list(df.columns)
# Data visualization
import matplotlib.pyplot as plt
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

plt.xlabel('sepal lenght (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='g',marker='*')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='b',marker='+')

plt.xlabel('petal lenght (cm)')
plt.ylabel('petal width (cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='g',marker='*')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='b',marker='+')

# Split train and test
from sklearn.model_selection import train_test_split
x1,x2,y1,y2=train_test_split(iris.data,iris.target,test_size=.15,random_state=10)

# Create a model
from sklearn.svm import SVC
svmc=SVC()
svmc.fit(x1,y1)

svmc.score(x2,y2)
