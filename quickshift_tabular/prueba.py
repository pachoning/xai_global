# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:05:51 2022

@author: delicado
"""

import numpy as np
import matplotlib.pyplot as plt 
from _quickshifttab import quickshifttab

#X = [np.random.normal(0, 1, 2) for i in range(100)] + [np.random.normal(5, 1, 2) for i in range(100)]

X1 = 15*(3+np.random.normal(0, 1, (100,2)))
X2 = 15*(3+np.random.normal(5, 1, (100,2)))

X =np.concatenate((X1,X2))

cl = [0] * 100 + [1] * 100

plt.scatter(X[:,0],X[:,1])
plt.show()

features = np.random.normal(0, .0001, (200,2))

X = np.concatenate((X,features),axis=1)



