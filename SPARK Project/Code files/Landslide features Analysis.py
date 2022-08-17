#!/usr/bin/env python
# coding: utf-8

# # Part 1- Landslide dataset Analysis
# 
# ## Project by Ajitesh Pandey under guidance of Professor Dr.Ritesh Kumar

# ### In this part of our project we will be seeing what all features influence landslides in Uttarakhand and how their Distribution looks like

# In[1]:


# Importing various libraries
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[2]:


# Import Data
df = pd.read_csv('Landslide_Dataset - Sheet1 (1).csv')


# In[3]:


df.head()


# In[5]:


df.info()


# ## Main causes of Landslide Trigger
# 
# ### 1-Continous Rainfall
# ### 2-FLooding of RIver
# ### 3-Heavy Rainfall
# ### 4-Anthrapogenic activity such as road cutting
# ### 5-Steep Slope with toe erosion by high river current
# ### 6-Cloud Burst

# In[8]:


sns.displot(df['Trigger and reason'])


# In[9]:


sns.barplot(x=df['Trigger and reason'],y=df['Landslide predictability'])


# ### So we see that heavy rainfall has the highest count but landslide occuring probability is highest in case of Continous rainfall.

# ## Now we will see the influence of Soil type and Material
# 
# ### 1-Debris
# ### 2-Rock
# ### 3-Rock cum Debris
# ### 4-Debris cum Earth
# ### 5-Earth cum Debris
# ### 6-Earth

# In[10]:


sns.displot(df['Material'])


# In[11]:


sns.barplot(x=df['Material'],y=df['Landslide predictability'],hue=df['Trigger and reason'])


# ### So we see that debris forms the major portion of soil cover followed by rock cum debris.We see in case of Debris Continous rainfall and Steep slope are primary trigger reasons of Landslides.

# ## Now we will see whether the particular location is active,reactivated site or a dormant one.
# 
# ### 1-Active
# ### 2-Reactivated
# ### 3-Dormant
# 

# In[12]:


sns.displot(df['Activity'])


# In[15]:


sns.barplot(x=df['Activity'],y=df['Landslide predictability'])


# ### So we conclude most of sites where Landslide has taken place are still while some have reactivated over a period of time.In both these sites occuring of Landslide predictability is much more than dormant region as shown by above graph.

# ## Now we will see the distribution of Geoscientific reason behinf the occurunce of previous landslide 
# ### 1- Presence of Loose and Unconsilated Material
# ### 2- Toe erosion fue to flooding and reduction in shear strength
# ### 3- Detachment of planes/wedge failure due to intersection of prominent joints
# ### 4- Heavy rainfall causing saturation of slope and reduction in shear stress

# In[16]:


sns.palplot(sns.color_palette("GnBu", 10))
sns.displot(df['Geoscientific reason'])


# In[19]:


sns.pointplot(x=df['Geoscientific reason'],y=df['Landslide predictability'],)


# In[21]:


sns.pointplot(x=df['Geoscientific reason'],y=df['Landslide predictability'],hue=df['Trigger and reason'])


# ### So we see presence of loose and unconsolidated material is the main reason but near the toe of rivers erosion due to landslide has the most probability.

# ## Now we will analyse the type of land Movements that occur and which of them are more dominant
# ### 1-Fall and Slide
# ### 2-Fall and Flow
# ### 3-Slide
# ### 4-Fall
# ### 5-Subsidence
# ### 6-Flow
# ### 7-Spread and Topple

# In[22]:


sns.displot(df['Movement'])


# In[26]:


import matplotlib.pyplot as plt
plt.plot(df['Movement'],df['Geoscientific reason'])
plt.xlabel('Movement')
plt.ylabel('Geoscientific reason')


# ### So we see that slides are the most common followed by falls.

# In[27]:


plt.plot(df['Landslide volume'])


# ## We see that most of the landslides which occured had value less in the range of 1000-10000 but there were some severe ones also with exceedingly high values

# ## Lets understand the cummulative rainfall distribution law

# In[28]:


plt.plot(df['Cummulative Rainfall'])


# In[29]:


plt.plot(df['Cummulative Rainfall'],df['Landslide predictability'])
plt.xlabel('Cummulative Rainfall')
plt.ylabel('Landslide predictability')


# ### The above graph conveys us two information-When cummulative rainfall tis greater than 70 mm there are more than 60 percent chances of a Landslide occuring and when it is less than 30 mm there are less than 20 percent chance of Landslide occuring with some exceptions .in mid range there is a lot of uncertainity and Intensity would play a deciding role.

# ## Now we analyse the last feature of our Dataset  -Rainfall Intensity which is one of the most important one as well.

# In[30]:


plt.plot(df['Rainfall Intensity'],df['Landslide predictability'])
plt.xlabel('Rainfall Intensity')
plt.ylabel('Landslide predictability')


# In[31]:


sns.barplot(x=df['Rainfall Intensity'],y=df['Landslide predictability'])


# In[34]:


sns.barplot(x=df['Rainfall Intensity'],y=df['Trigger and reason'])


# In[35]:


sns.barplot(x=df['Trigger and reason'],y=df['Rainfall Intensity'])


# ## So from above graphs we conclude that when heavy rainfall or cloud bursts were Landslide Triggers then intensity was more than 0.55
# 

# In[36]:


sns.barplot(x=df['Trigger and reason'],y=df['Cummulative Rainfall'])


# In[ ]:




