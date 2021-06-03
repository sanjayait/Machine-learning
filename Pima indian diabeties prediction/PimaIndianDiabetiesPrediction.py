# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 00:38:40 2021

@author: Sanjay
"""

# Import dependencies
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# dataframe
myarray = np.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pd.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)


# Load CSV using Pandas from URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
pd.set_option('display.max_columns', None)

# Explore data
data.head(10)
data.describe()
data.info()
data.isna().sum()
correlation = data.corr()


# Data visualization
data.groupby(['class']).mean()

for i, j in enumerate(names):
    plt.subplot(3, 3, i+1)
    plt.hist(data[j])
    plt.xlabel(j)
    plt.ylabel('count')
plt.show()


for i, j in enumerate(names):
    plt.subplot(3, 3, i+1)
    plt.bar(data['class'], data[j])
    plt.xlabel(j)
    plt.ylabel('count')
plt.show()

scatter_matrix(data, figsize=(16,10), grid=True, marker='^', color='g')


# Data Scaling
array = data.values

# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# summarize transformed data
np.set_printoptions(precision=3)
print(rescaledX[0:5,:])


# Split out validation dataset
xtrain,xtest,ytrain,ytest = train_test_split(X, Y, test_size=0.15, random_state=1)

# Create models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDR', LinearDiscriminantAnalysis()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NMN', MultinomialNB()))
models.append(('NBB', BernoulliNB()))
models.append(('RFC', RandomForestClassifier(n_estimators=200, max_features=3)))

results = []
modelNames = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    crossValResults = cross_val_score(model, xtrain, ytrain, cv=kfold, scoring='accuracy')
    modelNames.append(name)
    results.append(crossValResults)
    print(f"{name} : {crossValResults.mean()}")
    #print("Accuracy: %.3f%% (%.3f%%)" % (crossValResults.mean()*100.0, crossValResults.std()*100.0))


# Grid search for algorithm tuning
alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid.fit(xtrain, ytrain)
print(grid.best_score_)
print(grid.best_estimator_.alpha)


# Compare Algorithms
plt.boxplot(results, labels=modelNames)
plt.title('Algorithm Comparison')
plt.show()


# Make LDA predictions on validation dataset
model1 = LinearDiscriminantAnalysis()
model1.fit(xtrain, ytrain)
predictionsLDA = model1.predict(xtest)


# Evaluate predictions
print(accuracy_score(ytest, predictionsLDA))
print(confusion_matrix(ytest, predictionsLDA))
print(classification_report(ytest, predictionsLDA))


# Save model to disk
pickle.dump(model1, open('DiabetiesPrediction.sav', 'wb'))
