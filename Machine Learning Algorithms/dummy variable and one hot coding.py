# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:58:09 2021

@author: Sanjay
"""

import pandas as pd
import numpy as np

df=pd.read_excel("dummy.xlsx")
df

# Create dummy variable
dummies = pd.get_dummies(df.town)

# Concatinate with orginal
merged = pd.concat([df,dummies],axis="columns")

# Drop town column
final=merged.drop(["town","gwalior"],axis="columns")

# Create a Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(final[["area","bhind","indore"]],df["price"])

lr.coef_
lr.intercept_

# Prdiction
lr.predict([[2800,0,1]])

lr.predict([[3000,0,0]])

# Some visualization
import matplotlib.pyplot as plt 

%matplotlib inline
plt.scatter(final.area, final.price)
xlabel="Area"
ylabel="Price"

X=final.drop("price",axis=1)
y=final.price

lr.score(X,y)

# LabelEncoding from sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

dfle=df
dfle.town=le.fit_transform(dfle.town)
X1=dfle[["town","area"]]
y1=dfle.price

# One-hot encoding from sklearn
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categories=[])
ohedf=ohe.fit_transform(X1)
