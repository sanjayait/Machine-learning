# -*- coding: utf-8 -*-
"""
Created on Sat May  8 16:42:35 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd
import os
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Merge 12 months of sales data into a single csv file
files = [file for file in os.listdir('./Sales_Data')]

allFiles = pd.DataFrame()
for file in files:
    df = pd.read_csv('Sales_Data/'+ file)
    allFiles = pd.concat([allFiles, df])
    
allFiles.to_csv('AllMonthsData.csv', index=False)

# Check DataFrame
df = pd.read_csv('AllMonthsData.csv')

# Cleaning Data
df.dropna(axis=0, inplace=True)
df.head()
df.tail()
df2 = df.copy()

# Augmented data with additional columns
""" Add Month column """
df2['Month'] = df2['Order Date'].str[0:2]
df2 = df2.loc[df2['Month'] !='Or']

df2['Month'] =df2['Month'].astype('int32')
df2['Quantity Ordered'] =df2['Quantity Ordered'].astype('int32')
df2['Price Each'] =df2['Price Each'].astype('float32')
df2['Order ID'] =df2['Order ID'].astype('int32')

df2['Sales'] = df2['Quantity Ordered'] * df2['Price Each']
df2.info()

# Best Month for sales ? How much was earned that month?
df2.groupby('Month').sum()['Sales'] # Only Sales
results = df2.groupby('Month').sum() # For all Comparision

# Data Visualization
monthName = ('Jan', 'Feb', 'Mar', 'Apr', 'May','June', 'July','Aug', 'Sep', 'Oct', 'Nov', 'Dec')

plt.bar(monthName, results['Sales'])
plt.xticks(monthName)
plt.xlabel('Months')
plt.ylabel('Sales in USD ($)')
plt.show()


# Which city had the highest numbers of sales
""" Add a city column to DataFrame """
df3 = df2.copy()
df3['City'] = df3['Purchase Address'].apply(lambda x : x.split(',')[1] + ' ' + x.split(',')[2].split(' ')[1])

byCity = df3.groupby('City').sum()
cities = list(byCity.index.unique())

plt.bar(cities, byCity['Sales'])
plt.xticks(cities, rotation=90)
plt.xlabel('Cities')
plt.ylabel('Sales in USD ($)')
plt.show()


# What time should be display advertisement to maximize sales
df3['Date Time'] = pd.to_datetime(df3['Order Date'])
df3['Hour'] = df3['Date Time'].dt.hour
df3['Hour'].unique()

byHour = df3.groupby('Hour').count()
hours = list(byHour.index)

plt.plot(hours, byHour['Sales'])
plt.grid()
plt.xticks(hours)
plt.xlabel('Hour')
plt.ylabel('Sales in USD ($)')
plt.show()


# What product are most often sold together
duplDf = df3[df3['Order ID'].duplicated(keep=False)
duplDf['Grouped'] = duplDf.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))

df4 = duplDf[['Order ID', 'Grouped']].drop_duplicates()
productComb = df4.groupby('Grouped').count()
df4.count()


# Which product sold the most
product = df3.groupby('Product').sum()
product['Quantity Ordered']
products = list(product.index)

plt.bar(products, product['Quantity Ordered'])
plt.xticks(products, rotation=90)
plt.xlabel('Products')
plt.ylabel('Counts')
plt.show