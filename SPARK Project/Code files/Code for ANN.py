#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing various libraries
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[2]:


# Import Data
df = pd.read_csv('Landslide_ANN.csv')


# In[3]:


df.head()


# In[4]:


X=df[['Activity','Trigger and reason','Material','Movement','Hydrological','Landuse','Geoscientific reason','Landslide volume','Cummulative Rainfall','Rainfall Intensity']]


# In[5]:


y=df[['Landslide predictability']]


# In[6]:


X.shape,y.shape


# In[7]:


from keras import models
from keras import layers
from keras import optimizers


# In[8]:


network = models.Sequential()
network.add(layers.Dense(24, activation='relu', input_shape=(10,)))
network.add(layers.Dense(32, activation='relu'))
network.add(layers.Dense(1))


# In[13]:



pip install keras


# In[14]:


from keras import models
from keras import layers
from keras import optimizers


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:





# In[22]:





# In[23]:


accuracy = network.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# In[ ]:




