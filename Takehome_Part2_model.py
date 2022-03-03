import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

os.chdir(r"C:\Users\thuyd\OneDrive\Documents\Sally\Take_Home_Assignment")
df_train = pd.read_csv('adult_trdata.csv')
df_test = pd.read_csv('adult_test.csv')
df_train.head()

#changing columns names on train

df_train.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_yrs', 'marital_stat', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df_train.head()
print(df_train.shape[0])


# In[5]:


df_test.reset_index(inplace=True)


# In[6]:


#changing columns names on test

df_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_yrs', 'marital_stat', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df_test.head()


# In[7]:


#Replacing data in train dataset

df_train['education'] = df_train['education'].replace({'Preschool','1st-4th','5th-6th', '7th-8th'}, 'Elementary/Secondary', regex=True)
df_train['education'] = df_train['education'].replace({'9th','10th', '11th', '12th', 'HS-grad'}, 'High-School', regex=True)
df_train['education'] = df_train['education'].replace({'Masters', 'Doctorate'}, 'Advanced', regex=True)
df_train['education'] = df_train['education'].replace({'Prof-school', 'Assoc-acdm', 'Assoc-voc'}, 'Professional-School', regex=True)

df_train['workclass'] = df_train['workclass'].replace({'Self-emp-inc', 'Self-emp-not-inc'}, 'Self-Employed', regex=True)
df_train['workclass'] = df_train['workclass'].replace({'Local-gov', 'State-gov', 'Federal-gov'}, 'Gov-job', regex=True)
df_train['workclass'] = df_train['workclass'].replace({'Without-pay','Never-worked'}, 'Unemployed', regex=True)

df_train['marital_stat'] = df_train['marital_stat'].replace({'Never-married','Divorced', 'Separated', 'Widowed'}, 'Single', regex=True)
df_train['marital_stat'] = df_train['marital_stat'].replace({'Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'}, 'Married', regex=True)
df_train.head()


# In[8]:


#Replacing data in test dataset

df_test['education'] = df_test['education'].replace({'Preschool','1st-4th','5th-6th', '7th-8th'}, 'Elementary/Secondary', regex=True)
df_test['education'] = df_test['education'].replace({'9th','10th', '11th', '12th', 'HS-grad'}, 'High-School', regex=True)
df_test['education'] = df_test['education'].replace({'Masters', 'Doctorate'}, 'Advanced', regex=True)
df_test['education'] = df_test['education'].replace({'Prof-school', 'Assoc-acdm', 'Assoc-voc'}, 'Professional-School', regex=True)

df_test['workclass'] = df_test['workclass'].replace({'Self-emp-inc', 'Self-emp-not-inc'}, 'Self-Employed', regex=True)
df_test['workclass'] = df_test['workclass'].replace({'Local-gov', 'State-gov', 'Federal-gov'}, 'Gov-job', regex=True)
df_test['workclass'] = df_test['workclass'].replace({'Without-pay','Never-worked'}, 'Unemployed', regex=True)

df_test['marital_stat'] = df_test['marital_stat'].replace({'Never-married','Divorced', 'Separated', 'Widowed'}, 'Single', regex=True)
df_test['marital_stat'] = df_test['marital_stat'].replace({'Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'}, 'Married', regex=True)


# In[9]:


# Checking NA values

np.sum(df_train.isna())
np.sum(df_test.isna())

# No NA in both train and test


# In[10]:


#Convert <=50k to 0 and >50k to 1

df_train['income'] = df_train['income'].replace({'<=50K'}, 0, regex=True)
df_train['income'] = df_train['income'].replace({'>50K'}, 1, regex=True)
df_test['income'] = df_test['income'].replace({'<=50K'}, 0, regex=True)
df_test['income'] = df_test['income'].replace({'>50K'}, 1, regex=True)


# In[11]:


# identify "?" in workclass and occupation columns, replace it with "Others"

df_train = df_train.replace({'workclass':{r'\?':'Others'},'occupation':{r'\?':'Others'}, 'education':{r'\?':'Others'}, 'marital_stat':{r'\?':'Others'},'relationship':{r'\?':'Others'},'race':{r'\?':'Others'},
                       'sex':{r'\?':'Others'},'native_country':{r'\?':'Others'}})

df_test = df_test.replace({'workclass':{r'\?':'Others'},'occupation':{r'\?':'Others'}, 'education':{r'\?':'Others'}, 'marital_stat':{r'\?':'Others'},'relationship':{r'\?':'Others'},'race':{r'\?':'Others'},
                       'sex':{r'\?':'Others'},'native_country':{r'\?':'Others'}})


# In[12]:


# identify "?" in workclass and occupation columns, replace it with "Others"
df_train['workclass'] = df_train['workclass'].replace({r'\?'}, 'Others', regex=True)
df_test['workclass'] = df_test['workclass'].replace({r'\?'}, 'Others', regex=True)
df_train['occupation'] = df_train['occupation'].replace({r'\?'}, 'Others', regex=True)
df_test['occupation'] = df_test['occupation'].replace({r'\?'}, 'Others', regex=True)


# ### EXPLORATORY

# In[15]:


df_train = df_train.drop(['native_country'], axis = 1)
df_test = df_test.drop(['native_country'], axis = 1)

# ### MODEL FITTING

# In[21]:


# spling df_train to x_train, y_train and df_test to x_test, y_test

x_train_ori = df_train.drop(['income'], axis = 1)
y_train = df_train['income']

x_test_ori = df_test.drop(['income'], axis = 1)
y_test = df_test['income']
x_train_ori.head()


# In[22]:


# Getting dummy from x_train and x_test

x_train = pd.get_dummies(x_train_ori,columns = ['workclass', 'education', 'marital_stat', 'occupation', 'relationship', 'race', 'sex'],drop_first=True)
x_test = pd.get_dummies(x_test_ori,columns = ['workclass', 'education', 'marital_stat', 'occupation', 'relationship', 'race', 'sex'],drop_first=True)


# In[23]:


# Normalize the dataset
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_train_scaled
scaler.fit(x_test)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)
x_train_scaled


# ### RANDOM FOREST CLASSIFIER

# In[24]:


# Random Forest Classifier
RFC_Model = RandomForestClassifier(n_estimators = 100)
RFC_Model.fit(x_train_scaled,y_train)
RFC_Predict = RFC_Model.predict(x_test_scaled)
RFC_Accuracy = accuracy_score(y_test, RFC_Predict)
print("Accuracy: " + str(RFC_Accuracy))

# In[27]:

LR_Model = LogisticRegression()
LR_Model.fit(x_train_scaled, y_train)
LR_Predict = LR_Model.predict(x_test_scaled)
LR_Accuracy = accuracy_score(y_test, LR_Predict)
print("Model Accuracy: " + str(LR_Accuracy))

# # B) API ENDPOINT

import joblib

joblib.dump(LR_Model, "model.pkl")
LR_Model = joblib.load("model.pkl")

model_columns = list(x_train.columns)
joblib.dump(model_columns, 'model_columns.pkl')


# In[32]:
joblib.dump(scaler, "Normalize.pkl")
scaler = joblib.load("Normalize.pkl")


