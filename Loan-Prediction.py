
# coding: utf-8

# In[1]:


from math import *


# In[4]:


import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("C:/Users/bilas/Documents/Python/train.csv")


# In[5]:


df.head(10)


# In[6]:


df.describe()


# In[8]:


df['Property_Area'].value_counts()


# In[9]:


df['Credit_History']. value_counts()


# In[14]:


df['ApplicantIncome'].hist(bins=100)


# In[12]:


import matplotlib as plt


# In[15]:


df.boxplot(column='ApplicantIncome')


# In[16]:


df.boxplot(column='ApplicantIncome',by= 'Education')


# In[17]:


df['LoanAmount'].hist(bins=50)


# In[18]:


df.boxplot('LoanAmount',by='Gender')


# In[19]:


temp1 = df['Credit_History'].value_counts(ascending=True)


# In[20]:



temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())


# In[21]:


print 'Frequency Table for Credit History:' 


# In[22]:


print temp1


# In[23]:


print '\nProbility of getting loan for each Credit History class:' 


# In[24]:


print temp2


# In[25]:


import matplotlib.pyplot as plt


# In[26]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")


# In[27]:


temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[29]:


df.apply(lambda x:sum(x.isnull()),axis=0)


# In[31]:


df.apply(lambda x:sum(x.isnull()),axis=0)


# In[32]:


df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


# In[33]:


df['Self_Employed'].fillna('No',inplace=True)


# In[34]:


df['LoanAmount_log'] = np.log(df['LoanAmount'])


# In[35]:


df['LoanAmount_log'].hist(bins=20)


# In[36]:


df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log']= np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)


# In[41]:


#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# In[39]:


def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  

