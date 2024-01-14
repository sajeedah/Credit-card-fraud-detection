#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[13]:


df = pd.read_csv(r"C:\Users\Sajeed\OneDrive\Desktop\PROJECT\CREDIT CARD FRAUD DETECTION\creditcard.csv")


# In[29]:


df.head(25)


# In[9]:


df.shape


# In[10]:


df.size


# In[20]:


df.tail()


# In[21]:


# dataset inofromation
df.info


# In[22]:


# checking the number of missing values in each column
df.isnull().sum()


# In[23]:


# if there is null value in the data need to fill it


# In[24]:


# distribution of legit transaction and fraudlelnt transaction
df['Class'].value_counts()


# In[25]:


# seprating the data for analysis
legit= df[df.Class==0]
fraud = df[df.Class==1]


# In[26]:


print(legit.shape)
print(fraud.shape)


# In[28]:


## statistical mesures of the data
legit.Amount.describe()


# In[30]:


fraud.Amount.describe()


# In[34]:


# compare values for both transaction
df.groupby('Class').mean()


# In[35]:


## Under-Sampling


# Build a sample dataset containing similar distribution of normal transaction and fraudulent transactions

# number of Fradulent tansaction=492

# In[37]:


legit_sample= legit.sample(n=492)


# concatenating two Data-frames

# In[38]:


new_dataset=pd.concat([legit_sample,fraud],axis=0)


# In[39]:


new_dataset.head()


# In[40]:


new_dataset.info


# In[41]:


new_dataset.tail()


# In[42]:


new_dataset['Class'].value_counts()


# In[43]:


new_dataset.groupby('Class').mean()


# In[44]:


df.groupby('Class').mean()


# splitting the data into features and target

# In[48]:


X= new_dataset.drop(columns ='Class',axis=1)
Y= new_dataset['Class']
print(X)

print(Y)
# Split the data into training data and Testing data

# In[53]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)


# In[54]:


print(X.shape,X_train.shape, X_test.shape)


# In[55]:


#model Training


# In[58]:


model = LogisticRegression()


# In[60]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# # Model Evaluation

# Accuracy Score

# In[62]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[67]:


print('Accuracy on Training data:',training_data_accuracy)


# In[68]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[69]:


print('Accuracy on Test data:',test_data_accuracy)


# In[ ]:




