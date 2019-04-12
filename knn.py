# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:44:15 2019

@author: Administrator

KNN 算法实现
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
np.random.seed(123)

digits = load_iris()
x, y = digits.data, digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y)

class knn:
    
    def __init__(self):
        pass
    
    def fit(self, x, y):
        self.data = x
        self.targets = y
    
    def euclidean_distance(self, x):
        ''' compute the euclidean_distance '''
        if x.ndim == 1:
            l2 = np.sqrt(np.sum((self.data - x)**2, axis = 1))
            
        if x.ndim == 2:
            n_sample,_ = x.shape
            l2 = [np.sqrt(np.sum((self.data - x[i])**2, axis = 1)) for i in range(n_sample)]
            
        return np.array(l2)
    
    def predict(self, x, k=1):
        dists = self.euclidean_distance( x )
        if x.ndim == 1:
            if k==1:
                nn = np.argmin(dists)
                return self.targets[nn]
            else: 
               kNN = np.argsort(dists)[:k]
               ynn = self.targets[kNN]
               max_vote = max(ynn, key = list(ynn).count)
               return np.asarray(max_vote)
           
        if x.ndim == 2:
            kNN = np.argsort(dists)[:,:k]
            ynn = self.targets[kNN]
            if k==1:
                return np.asarray(ynn.T.tolist()[0])
            else:
                n_sample,_ = x.shape
                max_vote = [max(ynn[i], key = list(ynn[i]).count) for i in range(n_sample)]
                return np.asarray(max_vote)
               
                
KNN = knn()
KNN.fit(x_train,y_train)        

p_result = KNN.predict(x_test, k =5)
pd.crosstab(p_result, y_test, rownames = ['predict'], colnames = ['real'])












