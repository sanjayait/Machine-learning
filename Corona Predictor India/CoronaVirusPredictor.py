# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:38:06 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_excel("owid-covid-data.xlsx")
df.head()

# Prepare data
dfIndia = df.loc[df["location"]=="India"]

dfIndia1 = dfIndia.dropna(axis=1)

cols = list(dfIndia1.columns)

dfIndia2 = dfIndia1.drop(['iso_code','continent','location','date','total_cases_per_million','new_cases_per_million','population','population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_thousand','life_expectancy','human_development_index'], axis=1)

dfindia3 = pd.DataFrame(dfIndia2, columns=['total_cases', 'new_cases'], index=None)
final = dfindia3.copy()

final.insert(0, 'id', final.index -33801)

# Data Visualization
plt.plot(final.id, final.total_cases, color='r')
plt.plot(final.id, final.new_cases, color='b')

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
col = ['total_cases','new_cases']
final[col] = scaler.fit_transform(final[col])

plt.plot(final.id, final.total_cases, color='r')
plt.plot(final.id, final.new_cases, color='b')


# Split data set
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=7)
X = final.drop(['total_cases','new_cases'], axis=1)
Xpoly = poly.fit_transform(X)
Y = final.total_cases

lr = LinearRegression()

lr.fit(Xpoly, Y)

lr.score(Xpoly, Y)

yPred = lr.predict(Xpoly)

# Data visualization Actual vs Predicted values
plt.plot(final.id, Y, color='r')
plt.plot(final.id, yPred, color='b')

print(f" After 1 day : {lr.predict(poly.fit_transform([[444+30]]))/1000000} Millions")
