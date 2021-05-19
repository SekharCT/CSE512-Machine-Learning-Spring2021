import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.stats import mode


def KNNClassifier(X_train, y_train, X_test, k):
    distance_matrix = cdist(X_test,X_train, 'euclidean')
    Idxs = y_train[np.argsort(distance_matrix,axis = 1).ravel()].reshape(distance_matrix.shape)[:,:k]
    y_hat = mode(Idxs, axis = 1)[0].ravel()
    return y_hat, Idxs

