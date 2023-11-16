#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
import pandas as pd
from pymysql import connect 
pd.set_option('display.max_columns', None)


# In[31]:


survey_data = pd.read_csv(r"E:\Guvidatascience\Projects\Final_project\Employee_survey_data.csv")


# In[32]:


survey_data.head()


# In[33]:


survey_data.shape


# In[34]:


survey_data.isnull().sum()


# In[35]:


survey_data.dtypes


# In[36]:


general_data=pd.read_csv(r"E:\Guvidatascience\Projects\Final_project\Employee_general_data.csv")


# In[37]:


general_data.head()


# In[38]:


general_data.shape


# In[39]:


general_data.isnull().sum()


# In[40]:


general_data.dtypes


# In[41]:


manager_suvery_data=pd.read_csv(r"E:\Guvidatascience\Projects\Final_project\Employee_manager_survey_data.csv")


# In[42]:


manager_suvery_data.head()


# In[43]:


manager_suvery_data.shape


# In[44]:


manager_suvery_data.dtypes


# In[45]:


manager_suvery_data.isnull().sum()


# In[46]:


print(survey_data.shape ,  general_data.shape , manager_suvery_data.shape)
print(survey_data.columns.tolist())
print(manager_suvery_data.columns.tolist())


# In[47]:


merged_df=pd.merge(survey_data,manager_suvery_data,on='EmployeeID')
final_df=pd.merge(general_data,manager_suvery_data,on='EmployeeID')

if(len(final_df.columns.tolist())   == (len(general_data.columns.tolist()) + len(survey_data.columns.tolist()) + len(manager_suvery_data.columns.tolist())) - 2):
  print("OK")
  


# In[48]:


final_df.head()


# In[49]:


final_df.shape


# In[50]:


final_df.dtypes


# In[ ]:





# In[51]:


final_df.Over18


# In[52]:


final_df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)


# In[53]:


final_df


# In[54]:


final_df.to_csv('Employee Attrition Analysis.csv', index=False)


# In[55]:


final_df.dtypes


# In[56]:


final_df.drop(['DistanceFromHome', 'StockOptionLevel', 'TrainingTimesLastYear','YearsWithCurrManager'], axis=1, inplace=True)


# In[57]:


final_df


# In[58]:


final_df.to_csv('Attrition_Analysis.csv', index=False)

