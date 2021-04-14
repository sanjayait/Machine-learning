# Price = m * area + b
# m=slop/Gradient ; b= Intercept
# Line equation: Y = mX + C

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_excel("area_price.xlsx")
df

# To draw the plot area vs price
%matplotlib inline
plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.Area,df.Price,color="red",marker="*")
plt.plot(df["Area"],lr.predict(df[["Area"]]),color="blue")

# Create a linearRegression Object
lr=LinearRegression()

# Create a Model
lr.fit(df[["Area"]],df.Price)

# Predict values
lr.predict(np.array(5000).reshape(-1,1))

# Cofficient/Slop/Gradient
lr.coef_

# Intercept
lr.intercept_

# Check Calculation
y=92.06349206*5000+(-172142.8571428572)
# Here y=same as predicted value

lr.predict(df[["Area"]])

%matplotlib inline
plt.xlabel("Area")
plt.ylabel("Price")
plt.scatter(df.Area,df.Price,color="red",marker="*")
plt.plot(df["Area"],lr.predict(df[["Area"]]),color="blue")        
                            
                            
