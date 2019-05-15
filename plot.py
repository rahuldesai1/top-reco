#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import os
import tensorflow as tf
import sys

# In[38]:


def plot(json_path):
    if json_path is None:
        return  
    df = pd.read_json(json_path)
    results_se = []
    results_ba = []
    for i in df['Threshold Values'][0]:
        results_se.append(i['Signal Efficiency'])
        results_ba.append(i['Background Acceptance'])
    
    plt.plot(results_ba, results_se)
    plt.xlabel("Background Acceptance")
    plt.ylabel("Signal Efficiency")


# In[37]:


def calcArea(results_se, results_ba):    
    total_area = 0
    values = list(zip(results_se, results_ba))
    values.sort(key=lambda x: x[1])

    prev = values[0][1]
    for curr in values:
        total_area += (curr[1] - prev) * curr[0]
        prev = curr[1]

    total_area


# In[39]:

try:
    inp = sys.argv[1]
except Exception as e:
    inp = None
    print(e)
    print("You must enter a valid path!")

plot(inp)


# In[ ]:




