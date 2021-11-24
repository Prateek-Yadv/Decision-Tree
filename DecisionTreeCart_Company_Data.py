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


Sales=pd.read_csv('C:/Users/prate/Downloads/Assignment/Decision Trees/Company_Data.csv')
Sales.head()


# In[34]:


Sales.describe


# In[35]:


Sales.isnull().sum()


# In[36]:


Sales.shape


# In[37]:


Sales.Sales.mean()


# In[38]:


#converting "Sales" in categorical variable
Sales1=pd.Series(Sales['Sales'])
s=[]
for i in Sales1:
    if i>8:
        s.append('good')
    
    else:
        s.append('bad')

print(s)


# In[39]:


Sales1=pd.Series(s)
Sales1=pd.concat([Sales1,Sales],axis=1)
Sales1=Sales1.rename({0:'o|p'},axis=1)


# In[40]:


Sales1.head()


# In[41]:


from sklearn.preprocessing import LabelEncoder
label_encoder =LabelEncoder()
Sales1['ShelveLoc']= label_encoder.fit_transform(Sales1['ShelveLoc']) 
Sales1['Urban']= label_encoder.fit_transform(Sales1['Urban'])
Sales1['US']= label_encoder.fit_transform(Sales1['US'])
Sales1['o|p']=label_encoder.fit_transform(Sales1['o|p'])
Sales1


# In[45]:


x=Sales1.iloc[:,1:12]
y=Sales1.iloc[:,0]


# In[46]:


x.head()


# In[47]:


y.head()


# In[48]:


x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.1,random_state=400)


# ### Building Decision Tree Classifier using Entropy Criteria

# In[49]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)


# In[50]:


model.get_n_leaves()


# In[51]:


preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 


# In[52]:


preds


# In[53]:


pd.crosstab(y_test,preds) # getting the 2 way table to understand the correct and wrong predictions


# In[54]:


# Accuracy 
np.mean(preds==y_test)


# In[55]:


print(classification_report(preds,y_test))


# ### Building Decision Tree Classifier (CART) using Gini Criteria

# In[56]:


from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)


# In[57]:


model_gini.fit(x_train, y_train)


# In[58]:


#Prediction and computing the accuracy
pred=model.predict(x_test)
np.mean(preds==y_test)


# In[60]:


fn=['sales','CompPrice','Income','Advertising','Population',
 'Price',
 'ShelveLoc',
 'Age',
 'Education',
 'Urban',
 'US']
cn=['yes', 'no']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8,8), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[61]:


#PLot the decision tree
tree.plot_tree(model);


# In[ ]:




