#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("C:/Users/Bharath/Downloads/spam.csv",encoding='latin-1')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)


# In[8]:


df.head()


# In[9]:


df.columns = ['cat', 'text']
df['cat'] = df['cat'].map({'ham': 0, 'spam': 1})


# In[10]:


df.head()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df=df.drop_duplicates()


# In[15]:


df.reset_index(inplace = True, drop = True)


# In[16]:


x = df['text']
y = df['cat'].astype(int)


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=21,test_size=0.2)


# In[19]:


from sklearn.feature_extraction.text import TfidfVectorizer
feat = TfidfVectorizer()
x_train = feat.fit_transform(x_train)
x_test = feat.transform(x_test)


# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


# In[28]:


rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)


# In[32]:


confusion_matrix(y_test, y_pred)


# In[34]:


accuracy=accuracy_score(y_test,y_pred)*100
print("The Accuracy of Email Spam Detection id {:.2f}".format(accuracy))

