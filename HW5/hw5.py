# -*- coding: utf-8 -*-


import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt

def assignment(X, centroids):
  distance_matrix = distance.cdist(X, centroids, 'euclidean')
  indices = np.argmin(distance_matrix, axis = 1)
  return indices

def get_error(X, centroids, idxs):
  error = 0.0
  for i in range(len(X)):
    error+= np.sum(distance.cdist(X[i].reshape(1,-1), centroids[idxs[i]].reshape(1,-1),'euclidean')**2)
  return error

def k_means(X,k):
  errors = []
  random = np.random.permutation(len(X))
  centroids = X[random[:k]]
  idxs_old = assignment(X,centroids)
  idxs_new = np.empty(len(centroids))
  while (True):
    new_centroids = np.array([X[idxs_old==k].mean(axis=0) for k in range(len(centroids))])
    idxs_new = assignment(X, new_centroids)
    errors.append(get_error(X, new_centroids, idxs_new))
    if (np.array_equal(idxs_old, idxs_new)):
      break
    else:
      idxs_old = idxs_new

  return new_centroids, errors

# train_data = pd.read_csv('mnist_train_hw5.csv').to_numpy()

# train_data = pd.read_csv('mnist_train_hw5.csv').to_numpy()
# y_train = train_data[:,0]
# X_train = train_data[:,1:]

# test_data = pd.read_csv('mnist_test_hw5.csv').to_numpy()
# y_test = test_data[:,0]
# X_test = test_data[:,1:]

# X_train,X_test = X_train/255, X_test/255
# X_train = np.asarray(X_train, dtype = np.float32)
# X_test = np.asarray(X_test, dtype = np.float32)

def utility(k=10):
  centroids, errors = k_means(X_train, k)
  X = np.arange(0,len(errors), 1)
  plt.plot(X,errors)
  plt.xlabel('Iterations')
  plt.ylabel('Squared Error')
  plt.show()

  print('Final SSD is : '+ str(errors[-1]))

  test_data_ids = assignment(X_test, centroids)

  fig = plt.figure()
  for i in range(len(centroids)):
    plt.subplot(4,4, i+1)
    plt.imshow(np.reshape(centroids[i], (28,28)))
    plt.title(np.sum(test_data_ids == i))
    plt.axis('off')
  plt.tight_layout()
  plt.show()

