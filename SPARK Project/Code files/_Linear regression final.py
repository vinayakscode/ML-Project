#!/usr/bin/env python
# coding: utf-8

# # So Now from the previous three ML classification models we know that we have obtained a high accuracy in predicting whether there are chances of no Landslide, Small Landslides or Large Disastrous Landslides.
# 
# # Now using the regression models we will try to predict the probability of occuring of a Landslide 

# # Now let us try to understand how linear regression works

# ![1_N1-K-A43_98pYZ27fnupDA.jpeg](attachment:1_N1-K-A43_98pYZ27fnupDA.jpeg)

# # So linear Regression actually finds the best fit line .Since we have a little more complex variable we will also be including higher powers to get the best results.

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
from sklearn.linear_model import LogisticRegression


# In[2]:


# Import Data
df = pd.read_csv('Landslide_dataforlinearregression.csv')


# In[3]:


df.head()


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# # Now let's see how each variable is corelated to Landslide predictability

# In[5]:


df.corr()


# # Now activity and trigger are categorical variables.regressionmodel need only float values so they need to be converted into dummy variables.

# In[6]:


X=df[['Activity','Trigger']]
X = pd.get_dummies(data=X, drop_first=True)
X.head()


# In[7]:


df.head()


# In[ ]:





# In[8]:


df=pd.concat([df,X],axis=1)
df.head()


# # Now specyfing the learning amd Target variables

# In[9]:


X=df[['Landslide Volume','Cummulative Rainfall','Rainfall Intensity','Product','Activity_Dormant','Activity_Reactivated','Trigger_Heavy rain','Trigger_Steep slope','Trigger_continous rainfall','Trigger_road cutting','Trigger_flooding']]


# In[10]:


Y=df[['Landslide predictability']]


# In[11]:


from sklearn.model_selection import train_test_split


# # Spilling training and test data set in ratio of 75 and 25

# In[12]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25)


# In[13]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# # Fitting test and train data in model

# In[14]:


model.fit(x_train, y_train)


# # Now since we dont have the direct accuracy function in Linear regression we will check its performance by following parameters
# 
# ## 1.Coefficient of determination- The coefficient of determination or R squared method is the proportion of the variance in the dependent variable that is predicted from the independent variable.
# ## If R2 is equal to 0, then the dependent variable cannot be predicted from the independent variable.
# ## If R2 is equal to 1, then the dependent variable can be predicted from the independent variable without any error.

# In[15]:


r_sq = model.score(X, Y)
print('coefficient of determination:', r_sq)


# ### The mean squared error (MSE) tells you how close a regression line is to a set of points. It does this by taking the distances from the points to the regression line (these distances are the “errors”) and squaring them. The squaring is necessary to remove any negative signs. It also gives more weight to larger differences. It’s called the mean squared error as you’re finding the average of a set of errors. The lower the MSE, the better the forecast.

# In[18]:


# Mean error
print("Mean  error: %.2f" % np.mean((model.predict(x_test) - y_test) ))
# The mean squared error
print("Mean  square error: %.2f" % np.mean((model.predict(x_test) - y_test)**2 ))


# In[19]:


# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(x_test, y_test))


# In[20]:


#To retrieve the intercept:
print(model.intercept_)

#For retrieving the slope:
print(model.coef_)


# # Now we will be predicting from our model.this function is deployed to a website as well which we will see later

# In[21]:


y_pred = model.predict(X)
print('predicted response:', y_pred, sep='\n')


# In[ ]:





# In[22]:


predictions = model.predict(x_test)


# In[23]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(y_test, predictions)


# ## Now from the above plot it is cleary visible that predicted values are falling on linear on a polynomial line so our model will be making predictions very accurate.

# In[26]:


plt.hist(y_test - predictions)


# ## Now from the above graph we see that we have the most predicted variables with zero error and graph is also uniform so our model is doing pretty well.

# In[25]:


from sklearn import metrics
metrics.mean_absolute_error(y_test, predictions)


# # So when we calculate the mean absolute error it comes to be 5.75 and since predicatibilty lies from 0 to 100 percent we can say that our model will predict landslide value with a +/- of 5.75 .

# In[29]:


#These all tasks are to get our model as pickle file to be deployed in website


# In[30]:


import joblib


# In[31]:


joblib.dump(model,"Landslide_predict.pkl")


# In[ ]:




