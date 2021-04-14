# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:26:41 2021

@author: Sanjay
"""

# Import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# Read dataset
df1=pd.read_csv("banglore.csv")
df1.drop_duplicates(inplace=True)
col=list(df1.columns)

df1.groupby("area_type")['price'].agg('mean').plot(kind='bar',xlabel="area_type")

corelation=df1.corr()

df2=df1.drop(['area_type','availability','society','balcony'],axis=1)
df2['size'].replace(' Bedroom','',inplace=True)

# Data Cleaning
df2.isna().sum()
df3=df2.dropna()
df3['bhk']=df3['size'].apply(lambda x: int(x.split(' ')[0]))

df3[df3['bhk']>20]

df3.total_sqft.unique()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df3[~df3['total_sqft'].apply(is_float)]

def average_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    
df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(average_num)
df4['total_sqft'].unique()

df4[df4['total_sqft'].isna()]
df5=df4.dropna()

# Create a new column
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']

# Preprecessing on location
df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count')
len(location_stats[location_stats<=10])
location_less_10=location_stats[location_stats<=10]

# Transform column
df5.location=df5.location.apply(lambda x: 'other' if x in location_less_10 else x)


# Outlier detection and removal
df5[df5.total_sqft/df5.bhk<300].head()
df6=df5[~(df5.total_sqft/df5.bhk<300)]

df6.price_per_sqft.describe()

# Define a function to remove outlier
def remove_pps_outlier(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out

df7=remove_pps_outlier(df6)

# Data visualization
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location) & (df.bhk==2)]
    bhk3=df[(df.location==location) & (df.bhk==3)]
    #matplotlib.rcParams['figure','figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price_per_sqft,color='b',label='2 bhk',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price_per_sqft,color='g',label='3 bhk',s=50,marker='+')
    plt.xlabel("Total square feet area")
    plt.ylabel("price per square feet")
    plt.title(location)
    plt.legend()


plot_scatter_chart(df7,"Hebbal")

# Define function to remove outlier
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df8=remove_bhk_outliers(df7)

# data visualization
plot_scatter_chart(df8,"Hebbal")

# Feature bath
df8[df8.bath>10]
df8[df8.bath>df8.bhk+2]

dh9=df8[~(df8.bath>df8.bhk+2)]
df9=dh9.drop(['size','price_per_sqft'],axis=1)

# Create dummy variable
dummy=pd.get_dummies(df9.location)

df10=pd.concat([df9,dummy.drop('other',axis=1)],axis=1)
df10.drop('location',axis=1,inplace=True)

# split train and test
x = df10.drop('price',axis=1)
y = df10.price

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=10)

# Create a model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)

# K_fold validation
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

ss=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
cross_val_score(lr,x,y,cv=ss)

# GridsearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

# Define function
def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)

""" LinearRegression is best """

def predict_price(location,sqft,bath,bhk):
    loc_index=np.where(x.columns==location)[0][0]
    x1=np.zeros(len(x.columns))
    x1[0]=sqft
    x1[1]=bath
    x1[2]=bhk
    if loc_index>=0:
        x1[loc_index]=1
    return lr.predict([x1])[0]

predict_price('Hebbal',1000,2,2)

# Export model to pickel file
import pickle
with open('banglore_home_price_model.pickle','wb') as f:
    pickle.dump(lr,f)
    
# Export columns to JSON
import json
columns={
    'data_columns': [col.lower() for col in x.columns]}
with open('columns.json','w') as f:
    f.write(json.dumps(columns))