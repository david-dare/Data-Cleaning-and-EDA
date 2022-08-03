#!/usr/bin/env python
# coding: utf-8

# **File will show basic EDA techniques in Python and also basic data cleaning on dataset.
# 

# **Importing libraries and dependencies

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt


# **Importing, inspecting head of dataframe:

# In[2]:


df = pd.read_csv(r"C:\Users\User\Downloads\airlines_final.csv")
df.head()


# **Checking datatypes and shape of dataframe:
# 

# In[3]:


df.info()
df.shape


# **Unnamed 0: column is duplication of axis and thus can be dropped

# In[4]:


df = df.drop(columns="Unnamed: 0")


# id can act as unique identifier and can be set as the index:
# 

# In[5]:


df = df.set_index("id")


# **wait_min is only numeric field - so descriptive statistics can be generated
# 

# In[6]:


df["wait_min"].describe()

**Useful to run
value counts on other axis to get insight into data:
# In[7]:


df["airline"].value_counts(normalize=True).head(10)



# In[8]:


df["destination"].value_counts(normalize=True).head(10)


# In[9]:


df["boarding_area"].value_counts(normalize=True).head()


# **cleanliness, safety and satisfaction are currently stored as object - better practice to store as category:

# In[10]:


df["cleanliness"] = df["cleanliness"].astype("category")
df["safety"] = df["safety"].astype("category")
df["satisfaction"] = df["satisfaction"].astype("category")                


# **dept_time currently stored as object need to be stored as data

# In[11]:


df["dept_time"] = pd.to_datetime(df["dept_time"])


# **I'm interested to visualise wait_min to see possible presence of outliers
# 

# In[12]:


sns.boxplot(df["wait_min"])
plt.show()


# 
# Assign values from the 95th percentile for wait_min to outliers(+95%) by creating new column 'wait_min_modified'

# In[13]:


percentile_95 = np.quantile(df["wait_min"],.95)


# In[14]:


df["wait_min_modified"] = df["wait_min"]


# In[15]:


df["wait_min_modified"] = np.where(df["wait_min_modified"] > percentile_95,percentile_95,df["wait_min"])


# **Check entries for duplicates and typos in dest_region

# In[16]:


df["dest_region"].unique()


# **Replace 'eur' with 'Europe','EAST US' with 'East US', 'middle east' with 'Middle East' to clean this column

# In[17]:


df["dest_region"] = df["dest_region"].str.replace("eur","Europe")
df["dest_region"] = df["dest_region"].str.replace("EAST US","East US")
df["dest_region"] = df["dest_region"].str.replace("middle east","Middle East")


# **Check changes have been effective

# In[18]:


df["dest_region"].unique()


# **Create mapping from day of week to weekday/end split:

# In[19]:


wday_mappings = {"Monday":"weekday","Tuesday":"weekday","Wednesday":"weekday","Thursday":"weekday","Friday":"weekday","Saturday":"weekend","Sunday":"weekend"}


# In[20]:


df["weekday/weekday"] = df["day"].map(wday_mappings)


# **Remove leading/trailing spaces in 'dest_size' column.

# In[21]:


df["dest_size"] = df["dest_size"].str.strip()


# **Check changes have been effective:

# In[22]:


df["dest_size"].unique()

**Remove any duplicated rows

# In[ ]:




