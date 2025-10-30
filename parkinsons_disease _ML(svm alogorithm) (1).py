#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Data Collection and Analysis

# In[2]:


parkinsons = pd.read_csv(r"C:\Users\shaik\Downloads\parkinsons (1).csv")


# In[3]:


parkinsons.head(10)


# In[4]:


parkinsons.shape
     


# In[5]:


parkinsons.info()


# In[6]:


# checking for missing values in each column
parkinsons.isnull().sum()
     


# In[7]:


# getting some statistical measures about the data
parkinsons.describe()


# In[8]:


parkinsons.keys()


# In[9]:


# distribution of target Variable
parkinsons['status'].value_counts()


# 1 --> Parkinson's Disease Present
# 
# 0 --> Healthy

# checking status

# In[10]:


parkinsons['status'].unique()


# In[11]:


parkinsons['status'] = pd.to_numeric(parkinsons['status'], errors='coerce')


# In[12]:


parkinsons['status'].isnull().sum()


# In[13]:


unique_values = parkinsons['status'].unique()
print(unique_values)


# In[14]:


print(parkinsons.dtypes)


# In[15]:


# Exclude 'name' column from the calculation
parkinsons.drop('name', axis=1).groupby('status').mean()


# This helps to give a clear distinction between people who have Parkinson's and those who don't have it.

# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'parkinsons' is your DataFrame
# You can set the desired height value (e.g., 3)
sns.pairplot(parkinsons, hue='status', palette='Dark2', diag_kind='hist')

# Show the plot
plt.show()


# # Data Pre-Processing

# In[20]:


X = parkinsons.drop(columns=['name','status'], axis=1)
y = parkinsons['status']


# In[21]:


print(X)


# In[22]:


print(y)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# # Data Standardization

# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


scaler = StandardScaler()
scaler.fit(X_train)
     


# In[26]:


X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)


# In[27]:


print(X_train)


# # Support Vector Machine Model

# In[28]:


model = svm.SVC()


# In[29]:


model.fit(X_train,y_train)
     


# # Model Evaluation

# In[30]:


# accuracy score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
     


# In[31]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[32]:


# accuracy score on training data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
     


# In[33]:


print('Accuracy score of training data : ', training_data_accuracy)


# In[34]:


from sklearn.metrics import classification_report,confusion_matrix
     


# In[35]:


pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# # Hyper-Tuning the model

# In[36]:


from sklearn.model_selection import GridSearchCV


# In[37]:


param_grid={'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[1]:


grid=GridSearchCV(svm.SVC(),param_grid,verbose=3)


# In[39]:


grid.fit(X_train,y_train)


# In[40]:


grid.best_params_


# In[41]:


grid.best_estimator_
     


# In[42]:


grid_pred=grid.predict(X_test)


# In[43]:


print(confusion_matrix(y_test,grid_pred))
print('\n')
print(classification_report(y_test,grid_pred))
     


# We can see the GridSearch has very much improved our model.
# 
# Now, we can build a predictive model.

# # Building a Predictive System

# In[44]:


import warnings
warnings.filterwarnings('ignore')


# In[45]:


input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

#converting input data into numpy array
input_data_numpy=np.asarray(input_data)

#reshaping
input_data_reshaped = input_data_numpy.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("The person does not have Parkinson's Disease.")

else:
  print("The person has Parkinson's Disease.")


# In[46]:


input_data = (119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)

input_data_numpy=np.asarray(input_data)

input_data_reshaped = input_data_numpy.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("The person does not have Parkinson's Disease.")

else:
  print("The person has Parkinson's Disease.")


# In[47]:


input_data = (199.22800,209.51200,192.09100,0.00241,0.00001,0.00134,0.00138,0.00402,0.01015,0.08900,0.00504,0.00641,0.00762,0.01513,0.00167,30.94000,0.432439,0.742055,-7.682587,0.173319,2.103106,0.068501)

#converting input data into numpy array
input_data_numpy=np.asarray(input_data)

#reshaping
input_data_reshaped = input_data_numpy.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("The person does not have Parkinson's Disease.")

else:
  print("The person has Parkinson's Disease.")
     


# In[49]:


input_data = (120.267,137.244,114.82,0.00333,0.00003,0.00155,0.00202,0.00466,0.01608,0.14,0.00779,0.00937,0.01351,0.02337,0.00607,24.886,0.59604,0.764112,-5.634322,0.257682,1.854785,0.211756)

#converting input data into numpy array
input_data_numpy=np.asarray(input_data)

#reshaping
input_data_reshaped = input_data_numpy.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print("The person does not have Parkinson's Disease.")

else:
  print("The person has Parkinson's Disease.")


# In[ ]:




