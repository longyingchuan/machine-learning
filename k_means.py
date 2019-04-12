# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:57:23 2019

@author: Administrator

k-means 算法实现
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
np.random.seed(123)


class k_means:
    
    def __init__(self, n_cluster = 4):
        self.k = n_cluster
        
    def l2_distance(self,datapoint):
         dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis = 1))
         return dists
     
    def classify(self,datapoint):
         dists = self.l2_distance(datapoint)
         return np.argmin(dists)
    
    def fit(self, data):
        n_sample,_ = data.shape
        self.centers = np.array(random.sample(list(data), self.k))
        self.initial_centers = np.copy(self.centers)
        old_assign = None
        n_iters = 0
        
        while True:
            new_assigns = [self.classify(datapoint) for datapoint in data]
            if new_assigns == old_assign:
                print('training finished after {} iterations'.format(n_iters))
                return 
            old_assign = new_assigns
            n_iters += 1
            
            for id_ in range(self.k):
                points_idx = np.where(np.array(new_assigns) == id_)
                datapoint = data[points_idx]
                self.centers[id_] = datapoint.mean(axis = 0)
     
    def plot_clusters(self,data):
        plt.figure(figsize=(12,10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:,0], data[:,1], c = 'y')
        plt.scatter(self.centers[:,0], self.centers[:,1], c='r')
        plt.scatter(self.initial_centers[:,0], self.initial_centers[:,1], c='b')
        plt.show()
        
        
        
 ''' 测试 '''
x, y = make_blobs(centers=4, n_samples=1000)
K_means = k_means()
K_means.fit(x)
K_means.plot_clusters(x)

       
         
        
        