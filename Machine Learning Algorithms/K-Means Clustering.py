# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:25:38 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read data file
df=pd.read_excel("income.xlsx")

# Data visusalization
plt.scatter(df.Age,df.Income,color='g',marker='*')

# Clustering Dataset
km=KMeans(n_clusters=3)
y_pred=km.fit_predict(df[["Age","Income"]])

df["Cluster"]=y_pred

df0=df[df.Cluster==0]
df1=df[df.Cluster==1]
df2=df[df.Cluster==2]


plt.scatter(df0.Age,df0.Income,color='g',marker='*')
plt.scatter(df1.Age,df1.Income,color='r',marker='+')
plt.scatter(df2.Age,df2.Income,color='b',marker='^')
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df["Income_s"]=scaler.fit_transform(df[["Income"]])
df["Age_s"]=scaler.fit_transform(df[["Age"]])

plt.scatter(df0.Age_s,df0.Income_s,color='g',marker='*')
plt.scatter(df1.Age_s,df1.Income_s,color='r',marker='+')
plt.scatter(df2.Age_s,df2.Income_s,color='b',marker='^')
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()


# K-Means on scaled data
y_pred_scale=km.fit_predict(df[["Age_s","Income_s"]])
df["cluster2"]=y_pred_scale

df00=df[df.cluster2==0]
df11=df[df.cluster2==1]
df22=df[df.cluster2==2]

plt.scatter(df00.Age_s,df00.Income_s,color='g',marker='*',label="one")
plt.scatter(df11.Age_s,df11.Income_s,color='r',marker='+',label="two")
plt.scatter(df22.Age_s,df22.Income_s,color='b',marker='^',label="three")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="black",label="centroid")
plt.xlabel("Age")
plt.ylabel("Income")
plt.legend()

# Centroid of Clusters
km.cluster_centers_
km.inertia_


# Elbow method
krang=range(2,11)
sse=[] # Sum of square error(sse)
for i in krang:
    km1=KMeans(n_clusters=i)
    km1.fit(df[["Age_s","Income_s"]])
    sse.append(km1.inertia_) # to find sse "inertia_" is used

plt.plot(krang,sse)
