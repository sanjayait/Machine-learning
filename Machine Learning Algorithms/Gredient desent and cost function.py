# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 16:04:51 2021

@author: Sanjay
"""

# Mean Squared Error = mse ( Cost Function)
#  Root Mean Squared Error = rmse ( Cost Function)

# Gredient descent is an algprithms that finds best fit line
# for given dataset

import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000
    n=len(x)
    learning_rate = 0.03
    
    for i in range(iterations):
        y_predicted = m_curr*x + b_curr 
        cost=(1/n)*sum([val**2 for val in (y-y_predicted)])
        md=-(2/n)*sum(x*(y-y_predicted))
        bd=-(2/n)*sum((y-y_predicted))
        m_curr = m_curr-learning_rate*md
        b_curr = b_curr-learning_rate*bd
        print(f"m {m_curr} b {b_curr},cost {cost}, iteration {i}")
        
  

x=np.array([1,2,3,4,5,6,7,8,9])
y=np.array([5,7,9,11,13,15,17,19,21])

gradient_descent(x,y)