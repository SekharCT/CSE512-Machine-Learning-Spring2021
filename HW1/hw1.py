import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_mean_and_variance_feature(X,feature_index,y,label_value):
    shape = X.shape
    mean_and_variance_list = []
    for i in range(shape[0]):
        if (y[i] == label_value):
              mean_and_variance_list.append(X[i,feature_index])

    mean_and_variance_array = np.array(mean_and_variance_list)
    mean = np.mean(mean_and_variance_array)
    variance = np.var(mean_and_variance_array)

    return mean, variance 

def get_mean_and_variance(X,y):
    feature_length = len(X[0,:])
    
    mean_array_0 = np.zeros(feature_length)
    variance_array_0 = np.zeros(feature_length)  

    mean_array_1 = np.zeros(feature_length)
    variance_array_1 = np.zeros(feature_length)
    for feature_index in range(feature_length):
        mean_array_0[feature_index], variance_array_0[feature_index] = get_mean_and_variance_feature(X,feature_index,y,0)
        mean_array_1[feature_index], variance_array_1[feature_index] = get_mean_and_variance_feature(X,feature_index,y,1)

    return mean_array_0, variance_array_0, mean_array_1, variance_array_1  

def get_windows(a,b, window_size = 7):
    window_list = []
    for i in range(a.shape[0] - window_size + 1):
        window_list.append(a[i:i+window_size].tolist() + b[i:i+window_size].tolist())
    
    # converting list to 2D numpy array
    return np.array(window_list).reshape(-1,window_size*2)

def learn_reg_params(x,y):
    window_size = 7
    X = get_windows(x[1:-1],y[1:-1],window_size)
    Y = y[1:][window_size:]
    model = LinearRegression(normalize = True).fit(X, Y)
    return model.coef_, model.intercept_

