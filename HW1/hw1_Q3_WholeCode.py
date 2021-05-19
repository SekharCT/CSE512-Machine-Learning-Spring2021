#!/usr/bin/env python
# coding: utf-8

# In[58]:


# setting the path to my local folder
get_ipython().run_line_magic('cd', 'D:\\StonyBrook\\Study\\ML CSE512\\CSE512_Spring21_HW1\\CSE512_Spring21_HW1')


# In[59]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[60]:


def get_windows(a,b, window_size = 7):
    window_list = []
    for i in range(a.shape[0] - window_size + 1):
        window_list.append(a[i:i+window_size].tolist() + b[i:i+window_size].tolist())
    
    # converting list to 2D numpy array
    return np.array(window_list).reshape(-1,window_size*2)


# In[61]:


covid_time_series = pd.read_csv('covid19_time_series.csv')
covid_time_series.head()


# In[62]:


covid_time_series = covid_time_series.to_numpy()


# In[63]:


def learn_reg_params(x,y):
    window_size = 7
    X = get_windows(x[1:-1],y[1:-1],window_size)
    Y = y[1:][window_size:]
    model = LinearRegression(normalize = True).fit(X, Y)
    return model.coef_, model.intercept_


# In[64]:


learn_reg_params(covid_time_series[0],covid_time_series[1])


# In[65]:


window_size = 7
X = get_windows(covid_time_series[0,1:-1],covid_time_series[1,1:-1],window_size)
Y = covid_time_series[1,1:][window_size:]
model = LinearRegression(normalize = True).fit(X, Y)


# In[66]:


# visualising actual vs predicted values
x = covid_time_series[0,1:][window_size:].astype(int)
y_actual = covid_time_series[1,1:][window_size:].astype(int)
y_predicted = model.predict(X)


# In[67]:


plt.style.use('seaborn-whitegrid')
plt.plot(np.log2(x[-50:]), y_actual[-50:],'.',color='red', label = 'actual_y')
plt.plot(np.log2(x[-50:]), y_predicted[-50:],'.', color='skyblue', label= 'predicted_y')
plt.legend(loc="upper left")
plt.xlabel('Time ()')
plt.ylabel('No. of deaths')
plt.show()
#plt.savefig("A.png")


# In[68]:


print(y_predicted.shape)
print(y_actual.shape)
print(x.shape)
print(X.shape)
print(Y.shape)


# In[69]:


y_normal = y_actual - y_predicted
y_normal = np.sort(y_normal)
mean_y_normal = np.mean(y_normal)
variance_y_normal = np.var(y_normal)
y_data = stats.norm.pdf(y_normal,mean_y_normal,variance_y_normal)

print("Mean : ", mean_y_normal)
print("Variance : ", variance_y_normal)

fig = plt.figure(figsize=(20, 5))
plt.subplot(131)
plt.hist(y_normal,bins= 10)
plt.xlabel('Error Value ( y - y_hat)')

plt.subplot(132)
plt.plot(y_normal, y_data, color = 'blue')
plt.show()

plt.figure()
plt.subplot(111)
plt.hist(y_normal,bins =10)
plt.plot(y_normal,y_data,color = 'blue')
plt.title('Combined histogram and Gaussian curve')
plt.show()


# Yes, gaussian is a good approximation for errors. From the histogram, it can be seen that the errors gather around a value(~0) and as per Central Limit Theorem, it looks that the data(i.e. error values) has normal distribution. 

# In[70]:


print("Mean : ", mean_y_normal)
print("Variance : ", variance_y_normal)


# 
