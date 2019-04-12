# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:58:13 2019

@author: Administrator

决策树算法实现
"""

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random


'''
决策树算法总共共有三种特征筛选方法，其中C3.0使用gain，C4.5使用gain_rate,Cart使用gini系数，经过测试，gini系数的使用相对更准确
'''

def entropy(y):
    ''' 计算熵 '''
    a, b = np.unique(y, return_counts = True)
    c = np.asarray(y.shape)
    y_entropy = -np.dot(b.T/c,np.log2(b/c))
    return y_entropy


def cross_entropy(x, y):
    ''' 计算信息增益和信息增益率 '''
    y_entropy = entropy(y)
    b_entropy = entropy(x)
    x_entropy = []
    for i in np.unique(x):
        d = y[np.where(x==i)]
        e = np.asarray(d.shape)/np.asarray(y.shape)*entropy(d)
        x_entropy.append(e)
    gain = y_entropy - np.sum(x_entropy)
    gain_rate = (y_entropy - np.sum(x_entropy))/b_entropy
    return gain, gain_rate


def gini(x):
    ''' 计算基尼系数 '''
    a, b = np.unique(x, return_counts = True)
    c = np.asarray(x.shape)
    d = b/c
    return 1-np.dot(d.T,d)


def cross_gini(x, y):
    x_gini = []
    for i in np.unique(x):
        d = y[np.where(x==i)]
        f = y[np.where(x!=i)]
        e = np.asarray(d.shape)/np.asarray(y.shape)*gini(d)+(1-np.asarray(d.shape)/np.asarray(y.shape))*gini(f)
        x_gini.append(e)        
    return np.sum(x_gini)   


'''

决策树的剪枝分为预剪枝和后剪枝，以预防过拟合现象的发生

使用信息增益和信息增益率的决策树：
后剪枝：通过递归计算每个节点往上缩回，计算整体的正则化的损失函数的降少量，然后进行剪枝，直至任何一个节点的缩回不能导致损失函数减少为止
预剪枝：通过在训练之前事先规定决策树的深度，叶子节点的个数等来进行有计划的训练

使用gini的cart决策树的剪枝：
cart剪枝算法从完全生长的决策树的底端剪去一些子树，使决策树变小。
两步走：
1、cart树生成尽可能大的决策树；
2、设置损失函数正则项a为Inf;
3、自下而上的计算g(t),g(t)是损失函数减小的程度的度量，选取a和g(t)的最小值；
4、自下而上的访问内部节点t，如果有g(t)=a,进行剪枝，并对叶节点t以多数表决决定其类，得到数T，
5、最后通过交叉验证选取最好的数。

'''












