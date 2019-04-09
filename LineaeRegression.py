# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 23:00:25 2019

@author: cxt

线性回归代码实现
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
np.random.seed(123)

x = 2*np.random.rand(500,1)
y = 5+3*x+np.random.rand(500,1)
fig = plt.figure(figsize=(8,6))
plt.scatter(x,y)
plt.title('Dataset')
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y)


class LinearRegression:
    
    def __init__(self):
        pass
    
    def train_gradient_descent(self,x,y,learning_rate=0.01, n_iters=100):
        """
        Trains a linear regression model using gradient descent
        """
        # Step 0: Initialize the parameters
        n_samples, n_features = x.shape
        self.weights = np.zeros(shape=(n_features,1))
        self.bias=0
        costs=[]

        for i in range(n_iters):
            # Step 1: Compute a linear combination of the input features and weights
            y_predict = np.dot(x,self.weights) + self.bias
          
            # Step 2: Compute cost over training set
            cost = (1/n_samples)*np.sum((y_predict-y)**2)
            costs.append(cost)
            
            if i % 100 == 0:
                print("Cost at iteration {}:{}".format(i,cost))
                
            # Step 3: Compute the gradients
            dj_dw = (2/n_samples)*np.dot(x.T,(y_predict-y))
            dj_db = (2 / n_samples) * np.sum((y_predict - y))
            
            # Step 4: Update the parameters
            self.weights = self.weights - learning_rate*dj_dw
            self.bias = self.bias - learning_rate*dj_db
            
        return self.weights, self.bias, costs
        
        
    def train_normal_equation(self, x, y):
        """
        Trains a linear regression model using the normal equation
        """
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
        self.bias = 0
        
        return self.weights, self.bias
        
    def predict(self,x):
        return np.dot(x, self.weights) +self.bias


'''测试'''
regressor = LinearRegression()
w_trained, b_trained, costs  = regressor.train_gradient_descent(x_train, y_train, learning_rate = 0.005, n_iters =600)
fig = plt.figure(figsize = (8,6))
plt.plot(np.arange(600), costs)
plt.title("Development of cost during training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()


n_samples, _ = x_train.shape
n_samples_test, _ = x_test.shape

y_p_train = regressor.predict(x_train)
y_p_test = regressor.predict(x_test)

error_train =  (1 / n_samples) * np.sum((y_p_train - y_train) ** 2)
error_test =  (1 / n_samples_test) * np.sum((y_p_test - y_test) ** 2)

print("Error on training set: {}".format(error_train))
print("Error on test set: {}".format(error_test))


fig = plt.figure(figsize=(8,6))
plt.scatter(x_train, y_train)
plt.plot(x_test, y_p_test,'y')
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()





        
            
         
