#!/usr/bin/env python
# coding: utf-8

# In[28]:


# setting the path to my local folder
get_ipython().run_line_magic('cd', 'D:\\StonyBrook\\Study\\ML CSE512\\CSE512_Spring21_HW1\\CSE512_Spring21_HW1')


# In[29]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


# In[30]:


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


# In[31]:


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


# In[32]:


covid_data = pd.read_csv('covid19_metadata.csv')
covid_data.head()
print (covid_data)


# In[33]:


covid_data['gender'].loc[225]


# In[34]:


# changing gender from categorical to numerical 
covid_data['gender'] = np.where(covid_data['gender']== 'M', 0, 1)

covid_data['survival'] = np.where(covid_data['survival']== 'Y', 1, 0)

print(covid_data)


# In[35]:


covid_data = covid_data.to_numpy()

#type(covid_data) = numpy.ndarray


# In[36]:


X = covid_data[:,:2]
y = covid_data[:,2]

print (X)
print (y)


# In[37]:


mu0,var0,mu1,var1 = get_mean_and_variance(X,y)

print (mu0)
print (var0)
print (mu1)
print (var1)


# In[38]:


x_data = np.arange(-1000, 1000, 10)

feature_size = mu0.shape
feature = 0
 
y_data = stats.norm.pdf(x_data, mu0[feature], var0[feature])
plt.plot(x_data, y_data, color = 'black')

y_data = stats.norm.pdf(x_data, mu1[feature], var1[feature])
plt.plot(x_data, y_data, color = 'blue')

plt.title('Age feature')
plt.show()

feature = 1

x_data = np.arange(-1, 1.5, 0.0001)

y_data = stats.norm.pdf(x_data, mu0[feature], var0[feature])
plt.plot(x_data, y_data, color = 'black')

y_data = stats.norm.pdf(x_data, mu1[feature], var1[feature])
plt.plot(x_data, y_data, color = 'blue')

plt.title('Gender feature')
plt.show()


# 2.2.c :- Approximating gender by Gaussian curve is not a good idea. From the curve we observe we have negative values which is not possible in real life when we have two genders (0 and 1). Also, the decimals do not make any sense. The approximated data gives us physically impossible predictions with non-zero probability. Also, we already know that the population will lie under those two genders giving no additional information from the graph.

# In[ ]:




