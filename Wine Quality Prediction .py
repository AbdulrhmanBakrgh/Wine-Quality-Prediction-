#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip XGboost 


# In[3]:


pip install xgboost


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('/Users/bakr/Downloads/WineQT.csv')


# In[6]:


df


# In[7]:


df.head


# In[8]:


df.info()


# In[10]:


df.describe().T


# In[11]:


df.isnull().sum()


# In[14]:


df.hist(bins=20 , figsize=(10 , 10))
plt.show()


# In[15]:


plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[16]:


plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[17]:


df = df.drop('total sulfur dioxide', axis=1)


# In[18]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]


# In[19]:


df.replace({'white': 1, 'red': 0}, inplace=True)


# In[20]:


features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=40)

xtrain.shape, xtest.shape


# In[21]:


norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)


# In[22]:


models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for i in range(3):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy : ', metrics.roc_auc_score(
		ytest, models[i].predict(xtest)))
	print()


# In[34]:


print(metrics.classification_report(ytest,
									models[1].predict(xtest)))



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




