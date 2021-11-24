#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report


# In[33]:


fraud=pd.read_csv('C:/Users/prate/Downloads/Assignment/Decision Trees/Fraud_check.csv')
fraud.head()


# In[34]:


fraud.describe


# In[35]:


fraud.isnull().sum()


# In[36]:


fraud.shape


# In[37]:


fraud=fraud.rename({'Marital.status':'Marital_status','Taxable.Income':'Taxable_Income','City.Population':'City_Population','Work.Experience':'Work_Experience'}
                  ,axis=1)


# In[38]:


fraud_df=pd.Series(fraud['Taxable_Income'])
sf=[]
for i in fraud_df:
    if i<=30000:
        sf.append('risky')
    else:
        sf.append('good')
print(sf)


# In[39]:


fraud_df=pd.DataFrame(sf)
fraud_df=pd.concat([fraud_df,fraud],axis=1)
fraud_df=fraud_df.rename({0:'o/p'},axis=1)
fraud_df


# In[40]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

fraud_df.iloc[:,0] = labelencoder.fit_transform(fraud_df.iloc[:,0])
fraud_df.iloc[:,1]=labelencoder.fit_transform(fraud_df.iloc[:,1])
fraud_df.iloc[:,2]=labelencoder.fit_transform(fraud_df.iloc[:,2])
fraud_df.iloc[:,-1]=labelencoder.fit_transform(fraud_df.iloc[:,-1])


# In[41]:


x=fraud_df.iloc[:,1:7]
y=fraud_df.iloc[:,0]


# In[42]:


x.head()


# In[43]:


y.head()


# In[44]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.1,random_state=400)


# ### Building Decision Tree Classifier using Entropy Criteria

# In[45]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[46]:


model.get_n_leaves()


# In[47]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[48]:


preds


# In[49]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[50]:


# Accuracy 
np.mean(preds==y_test)


# In[51]:


print(classification_report(preds,y_test))


# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[52]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[53]:


model_gini.fit(x_train, y_train)


# In[54]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# In[55]:


fn=['Undergrad',
 'Marital_Status',
 'Taxable_Income',
 'City.Population',
 'Work_Experience',
 'Urban']
cn=['yes','no']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[56]:


#PLot the decision tree
tree.plot_tree(model);


# In[ ]:




