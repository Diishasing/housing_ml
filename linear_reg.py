#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# Lets load the boston house pricing prediction

# In[ ]:


from sklearn.datasets import fetch_california_housing


# In[3]:


housing = fetch_california_housing()
housing.keys()


# In[4]:


# print(housing.DESCR)


# # In[5]:


# print(housing.data)


# # In[6]:


# print(housing.target)


# # In[7]:


# print(housing.feature_names)


# Preparing the dataset

# In[8]:


dataset = pd.DataFrame(housing.data, columns = housing.feature_names)
dataset.head()


# # In[9]:


# dataset.info()


# # In[10]:


# #summarizing the stats of the data
# dataset.describe()


# In[ ]:


# #check the missing values
# dataset.isnull()


# In[11]:


#Exploratory data analysis

# #correlation
# dataset.corr()


# # In[12]:


# import seaborn as sns
# sns.pairplot(dataset)


# In[13]:


# plt.scatter(dataset['AveBedrms'], dataset['MedInc'])


# # In[14]:


# sns.regplot(x = 'AveBedrms', y = 'MedInc', data=dataset )


# In[15]:


dataset['Price'] = housing.target
dataset.head()


# In[16]:


x = dataset.iloc[:,: -1]
y = dataset.iloc[:, -1]


# In[17]:


print(y)


# In[18]:


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[19]:


x_test


# In[20]:


#standardize the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[21]:


x_train = scaler.fit_transform(x_train)


# In[22]:


x_test = scaler.transform(x_test)


# In[ ]:


import pickle
pickle.dump(scaler, open('Scaler.pkl', 'wb') )


# In[23]:


# x_test


# # Model training

# # In[24]:


# from sklearn.linear_model import LinearRegression


# # In[25]:


# regressor = LinearRegression()


# # In[26]:


# regressor.fit(x_train, y_train)


# # In[27]:


# print(regressor.coef_)


# # In[28]:


# print(regressor.intercept_)


# # In[29]:


# ##on which parameters the model has been trained
# regressor.get_params()


# # In[30]:


# ##prediction with the test data
# reg_pred = regressor.predict(x_test)
# reg_pred


# # In[31]:


# ##plot the scatter plot for the prediction
# plt.scatter(y_test, reg_pred)


# # In[32]:


# ## prediction with the residual(error)
# residual = y_test - reg_pred
# residual


# # In[33]:


# ##plotting the residuals
# sns.displot(residual, kind = 'kde')
# #normal distributing llike this below is a good thing


# # In[34]:


# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error

# print(mean_absolute_error(y_test, reg_pred))
# print(mean_squared_error(y_test, reg_pred))
# print(np.sqrt(mean_squared_error(y_test, reg_pred)))


# # In[37]:


# #R-Square and adjusted R square
# from sklearn.metrics import r2_score
# score = r2_score(y_test, reg_pred)
# score


# # In[38]:


# #adjusted r-square
# 1 - (1 - score)*(len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)


# # In[41]:


# #new data prediction
# ##dont forget to do the standardization of the data
# housing.data[0].reshape(1, -1)


# # In[44]:


# scaler.transform(housing.data[0].reshape(1, -1)) #standardizing


# # In[45]:


# regressor.predict(scaler.transform(housing.data[0].reshape(1, -1)))


# # In[47]:


# #pickle the model file for deployment

# import pickle


# # In[51]:


# pickle.dump(regressor, open('regmodel.pkl', 'wb'))


# # In[52]:


# #loading the pickle model
# pickled_model = pickle.load(open('regmodel.pkl', 'rb'))


# # In[ ]:




