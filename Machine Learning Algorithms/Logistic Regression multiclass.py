# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:44:26 2021

@author: Sanjay
"""

# Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits=load_digits()
digits.data[0]

plt.gray()
plt.matshow(digits.images[99])
plt.show()

digits.target[0:25]

# Split train test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(digits.data,digits.target,test_size=0.2,random_state=10)

# Create a Model
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(xtrain,ytrain)

# Check score
lor.score(xtest,ytest)

digits.target[99]

lor.predict([digits.data[99]])

# confusion matrix
y_predicted=lor.predict(xtest)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(ytest,y_predicted)


# EXERCISE
from sklearn.datasets import load_iris
data=load_iris()

data.data[0]
x1,y1,x2,y2=train_test_split(data.data,data.target,test_size=0.2,random_state=10)

lor2=LogisticRegression()
lor2.fit(x1,x2)

lor2.score(y1,y2)
lor2.predict([[5.1, 3.5, 1.4, 0.2]])

data.target_names[0]

y_pred2=lor2.predict(y1)
cm2=confusion_matrix(y2,y_pred2)
