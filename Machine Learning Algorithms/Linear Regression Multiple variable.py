# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:32:55 2021

@author: Sanjay
"""
# Prediction find on
# 1- 3000sqft, 3 bedrooms, 40 years old
# 1- 2500sqft, 4 bedrooms, 5 years old

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Price = m1*area + m2*bedrooms + m3*age + b
# m = coef/slop/gradient
# b = intercept
# y = m1x + m2x + m3x + b

# Read file
df=pd.read_excel("area_price2.xlsx")
df

df.bedrooms.median()
df.bedrooms.mean()

import math
median_bedrooms=math.floor(df.bedrooms.median())

df.bedrooms.fillna(median_bedrooms,inplace=True)

# Create linerRegression class object
lr=LinearRegression()
lr.fit(df[["area","bedrooms","age"]],df["price"])

# Cofficient
lr.coef_

# Intercept
lr.intercept_

lr.predict(np.array([5000,4,5]).reshape(1,-1))
