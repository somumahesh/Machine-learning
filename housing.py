#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_excel('1553768847_housing.xlsx')


# In[3]:


df.head()


# In[ ]:





# In[35]:


X=df.iloc[:,0:9]


# In[36]:


X.head()


# In[37]:


Y=df.iloc[:,9]


# In[38]:


Y.head()


# In[8]:


pd.isnull(X)


# In[9]:


X.tail()


# In[10]:


df.isnull().sum()


# In[39]:


df[df['total_bedrooms'].isnull()]


# In[40]:


df["total_bedrooms"]=df["total_bedrooms"].fillna(value=df["total_bedrooms"].mean())


# In[47]:


X['ocean_proximity']=X['ocean_proximity'].replace(['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'],[0,1,2,3,4
                                                                                                              ])


# In[48]:


X.head()


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


# In[77]:


x_train.head()


# In[78]:


from sklearn.preprocessing import StandardScaler


# In[79]:


sc= StandardScaler()


# In[80]:


x_train.iloc[:,0:]=sc.fit_transform(x_train.iloc[:,0:])


# In[81]:


x_train.head()


# In[82]:


x_test.iloc[:,0:]=sc.fit_transform(x_test.iloc[:,0:])


# In[83]:


x_test.head()


# In[84]:


from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[85]:


model = LinearRegression()


# In[86]:


model.fit(x_train,y_train)


# In[87]:


pred=model.predict(x_test)





# In[88]:


from sklearn.metrics import mean_squared_error


# In[89]:


mean_squared_error(y_test,pred)


# In[90]:


y_test


# In[91]:


pred


# In[92]:


from sklearn.tree import DecisionTreeRegressor


# In[93]:


model1=DecisionTreeRegressor()


# In[94]:


model1.fit(x_train,y_train)


# In[95]:


pred1=model1.predict(x_test)


# In[96]:


mean_squared_error(y_test,pred1)


# In[97]:


from sklearn.ensemble import RandomForestRegressor


# In[98]:


model2=RandomForestRegressor()


# In[99]:


model2.fit(x_train,y_train)


# In[100]:


pred2=model2.predict(x_test)


# In[101]:


mean_squared_error(y_test,pred2)


# In[ ]:




