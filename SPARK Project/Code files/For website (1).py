#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('Web_model1.csv')


# In[3]:


data.head(3)


# In[4]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[5]:


x = data[['Cummulative_rainfall', 'Intensity', 'Product']]
y = data['Landslide Predictability']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[7]:


from sklearn.linear_model import LinearRegression
classifier=LinearRegression()
classifier.fit(X_train, y_train)


# In[8]:


from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error


# In[9]:


predictions = classifier.predict(X_test)
print(f'R^2 score: {r2_score(y_true=y_test, y_pred=predictions):.2f}')
print(f'MAE score: {mean_absolute_error(y_true=y_test, y_pred=predictions):.2f}')
print(f'EVS score: {explained_variance_score(y_true=y_test, y_pred=predictions):.2f}')


# In[10]:


import pickle
with open('Landslide_predict1.pkl', 'wb') as file:
 pickle.dump(classifier, file)


# In[ ]:




