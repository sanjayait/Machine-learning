# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 17:31:25 2021

@author: Sanjay
"""

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pickle


# Load Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Summarize dataset
dataset.shape
dataset.head(10)
dataset.describe()

# class distribution
print(dataset.groupby('class').size())

""" Data Visualization """
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Histogram Plots
dataset.plot(kind='hist', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Line Plots
dataset.plot(kind='line', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

dataset.hist()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.15, random_state=1)


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# Evaluate each model in turn
results = []
modelNames = []
for name, model in models:
    kFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    crossVal = cross_val_score(model, X, y, cv=kFold, scoring='accuracy')
    results.append(crossVal)
    modelNames.append(name)
    print(f"{name} : {crossVal.mean()} : {crossVal.std()}")


# Compare Algorithms
pyplot.boxplot(results, labels=modelNames)
pyplot.title('Algorithm Comparison')
pyplot.show()


# Make SVM predictions on validation dataset
model1 = SVC(gamma='auto')
model1.fit(X_train, Y_train)
predictionsSVM = model1.predict(X_validation)


# Make LDA predictions on validation dataset
model2 = LinearDiscriminantAnalysis()
model2.fit(X_train, Y_train)
predictionsLDA = model2.predict(X_validation)


# Evaluate predictions
print(accuracy_score(Y_validation, predictionsSVM))
print(confusion_matrix(Y_validation, predictionsSVM))
print(classification_report(Y_validation, predictionsSVM))


# Evaluate predictions
print(accuracy_score(Y_validation, predictionsLDA))
print(confusion_matrix(Y_validation, predictionsLDA))
print(classification_report(Y_validation, predictionsLDA))


# save the model to disk
filename = 'finalized_iris_model.sav'
pickle.dump(model2, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_validation, Y_validation)
print(result)