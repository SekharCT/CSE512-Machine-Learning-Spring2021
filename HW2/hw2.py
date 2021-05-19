import numpy as np
import random 
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay


def logreg_predict_prob(W, b, X):
    theta_transpose = np.append(W,b.reshape(-1,1), axis = 1)  # reshape b if necessary b.reshape(-1,1)
    
    # theta_transpose ( k-1 x d+1)
    # append 1 to X -> X_bar (n x d+1)
    X_bar  = np.append(X, np.ones((len(X),1)), axis = 1)
    
    P = np.zeros((len(X),len(theta_transpose) + 1)) # n x k
    
    for i in range(len(X)): # 1 to n
        A = np.zeros(len(theta_transpose) + 1) # k
        for j in range(len(W)): # 1 to k-1
            # P (Y =j|X; theta_transpose) 
            A[j] = np.dot(theta_transpose[j],X_bar[i])
        A = np.exp(A - np.max(A))
        A = A/np.sum(A)
        P[i] = A
    
    return P


def logreg_predict_class(W, b, X):
    P = logreg_predict_prob(W, b, X)
    y_hat = P.argmax(axis = 1).reshape(-1,1)
    return y_hat

def permute(end):
    array = np.arange(end)
    random.shuffle(array)
    return array.astype(np.int)

def separate_theta_W_b(theta): # theta is k-1 x d+1
    W = theta[:,:-1]
    b = theta[:,-1].reshape(-1,1)
    return W,b

def loss_theta(X,y,theta):
    W,b = separate_theta_W_b(theta)
    P = logreg_predict_prob(W,b,X)
    loss = 0.0 
    for i in range(len(X)):
            loss+= np.log(P[i,y[i]])

    return (-1/len(X)) * loss

def gradient_descent(X,y,batch,theta):
    
    W,b = separate_theta_W_b(theta)
    P = logreg_predict_prob(W,b,X)
    result = np.zeros(theta.shape)
    for i in range(len(theta)):
        descent = np.zeros(X.shape[1] + 1) 
        for j in batch:
            if(y[j] == i):
                descent = descent +  (1 - P[j,i]) * (np.append(X[j],[1]))
            else : 
                descent = descent + (0 - P[j,i]) * (np.append(X[j],[1]))
        
        result[i] = descent 
                             
    return result* (-1/len(batch))



def logreg_fit(X, y, m, eta_start, eta_end, epsilon, max_epoch = 1000):
    # permute
    array = permute(len(X))
    
    # dividing into batches
    batches = np.array_split(array,m)
    
    # obtain k-value
    k = np.max(y) + 1 
    d = X.shape[1] # no. of features
    
    theta = np.zeros((k-1,d+1))
    Loss_list = []
    eta = eta_start
    
    for epoch in range(max_epoch):
        theta_old = theta
        for batch in batches:
            dL_dtheta = gradient_descent(X,y,batch,theta)
            # update theta
            theta = theta - eta*dL_dtheta
        
        loss_theta_old = loss_theta(X,y,theta_old)
        loss_theta_current = loss_theta(X,y,theta) 
        Loss_list.append(loss_theta_current)
        
        if( (loss_theta_old - loss_theta_current) < (epsilon * loss_theta_old) ):
            eta = eta/10
        if(eta < eta_end):
            break
        print("\nCompleted epoch: " + str(epoch))
    W,b = separate_theta_W_b(theta)
    Loss_list = np.array(Loss_list)
    return W,b

'''
W,b = logreg_fit(X_train, y_train, 256, 0.01, 0.00001, 0.0001,10)

'''