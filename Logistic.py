# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:16:00 2019

@author: cxt

逻辑回归实现

"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_blobs
np.random.seed(123)


class logistic:
    
    def __init__(self):
        pass
    
    def sigmod(self, a):
        return  1/(1 + np.exp(-a))
    
    def train(self, x, y, learning_rate, iter_num):
        '''
        train logistic model
        '''
        n_sample, n_features = x.shape
        self.weight = np.zeros(shape = (n_features,1))
        self.bias = 0
        costs = []
        
        for i in range(iter_num):
            y_predict = self.sigmod(np.dot(x, self.weight) + self.bias)
            wd = (1 / n_sample) * np.dot(x.T, y_predict - y)
            bd = (1 / n_sample) * np.sum(y_predict - y)
            self.weight =  self.weight - learning_rate*wd
            self.biaS = self.bias - learning_rate*bd
            cost = -(1 / n_sample)*np.sum(y*np.log(y_predict)+(1 - y)*np.log(1 - y_predict))
            costs.append(cost)
            
        return self.weight,self.bias,costs
        
    def predict(self, x):
        '''
        preict the value using train
        '''
        return sigmod(np.dot(x, self.weight) + self.bias)
    

'''
使用随机数据测试数据
'''

x, y = make_blobs(n_samples= 1000, centers=2)
# Reshape targets to get column vector with shape (n_samples, 1)
y = y[:, np.newaxis]
# Split the data into a training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y) 
  
Logistic = logistic()

w_trained, b_trained, costs = Logistic.train(x_train, y_train, learning_rate = 0.009, iter_num=600)

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(600), costs,'b')
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

y_predict = Logistic.predict(x_train)
fig1 = plt.figure(figsize = (8,6))
ax1 = fig1.add_subplot(111)
ax1.scatter(np.dot(x_train,w_trained)+b_trained, y_train, c = 'b')
ax1.scatter(np.dot(x_train,w_trained)+b_trained, y_predict, c ='r')
plt.show()






















    
    
