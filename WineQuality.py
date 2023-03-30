#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as npp
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


wine = pd.read_csv("wineqt.csv")


# In[6]:


wine.head()


# In[7]:


wine.tail()


# In[13]:


wine.shape


# In[14]:


wine.isnull().sum()


# In[15]:


wine.dropna(inplace=True)
#wine.update(wine.fillnawine.man())


# In[18]:


wine['type'].value_counts(normalize=True)


# In[19]:


wine['quality'].value_counts(normalize=True)


# In[20]:


sns.countplot(x="type",data=wine)


# In[23]:


plt.figure(2)
plt.subplot(121)
sns.distplot(wine['fixed acidity'])
plt.subplot(122)
wine['fixed acidity'].plot.box(figsize=(15,5))


# In[24]:


plt.figure(figsize=(10,7))
sns.barplot(x='quality',y='fixed acidity',data=wine)


# In[25]:


sns.pairplot(wine)


# In[26]:


wine.corr()


# In[27]:


plt.figure(figsize=(15,10))
sns.heatmap(wine.corr(),cmap='coolwarm')


# In[28]:


wine_new=wine.drop('total sulfur dioxide' ,axis=1)


# In[29]:


wine_new.head()


# In[31]:


wine_ml =pd.get_dummies(wine_new,drop_first=True)


# In[32]:


wine_ml.head()


# In[33]:


wine_ml.isnull().sum()


# In[35]:


Y = wine_ml['quality'].apply(lambda y:1 if y>7 else 0)
Y


# In[39]:


X = wine_ml.drop('quality',axis=1)


# In[40]:


from sklearn.preprocessing import StandardScaler


# In[44]:


scaler=StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)


# In[79]:


pred_test = np.asarray(pred_test).reshape(1,-1)
scaler.fit(pred_test)
pred_test_std = scaler.transform(pred_test)


# In[80]:


X=X_standard


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.2,random_state=245)


# In[83]:


from sklearn.linear_model import LogisticRegression


# In[51]:


logreg=LogisticRegression()
logreg.fit(X_train,Y_train)


# In[52]:


y_pred=logreg.predict(X_test)


# In[84]:


y_pred_test_output=logreg.predict(pred_test_std)
y_pred_test_output


# In[54]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[55]:


accuracy_score(Y_test,y_pred)


# In[56]:


print(classification_report(Y_test,y_pred))


# In[58]:


print(confusion_matrix(Y_test,y_pred))


# In[59]:


from sklearn.ensemble  import RandomForestClassifier


# In[66]:


rfc=RandomForestClassifier()
rfc.fit(X_train,Y_train)


# In[67]:


rfc_pred=rfc.predict(X_test)


# In[68]:


accuracy_score(Y_test,rfc_pred)


# In[69]:


rfc.feature_importances_


# In[71]:


pd.Series(rfc.feature_importances_,index=wine_ml.drop('quality',axis=1).columns).plot(kind='barh')


# In[72]:


pred_test=wine.iloc[2]
pred_test


# In[74]:


pred_test['type']=1
pred_test.drop(['quality','total sulfur dioxide'],inplace=True)
pred_test


# In[ ]:




