#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

import warnings
warnings.simplefilter(action='ignore')

import timeit

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters


# In[2]:


CO_timeseries_unpivot_path = '/home/ubuntu/workspace_rohan/project/data/processed/CO_timeseries_unpivot.csv'


# In[ ]:


# data_tseries_unpivot = pd.read_csv(CO_timeseries_unpivot_path)


# In[ ]:


# data_tseries_unpivot_10 = data_tseries_unpivot.head(190)


# In[ ]:


# data_tseries_unpivot_10 = data_tseries_unpivot_10[['index', 'week_int', 'week_value']]


# In[ ]:


# data_tseries_unpivot_10.tail()


# In[ ]:


# data_tseries_unpivot = data_tseries_unpivot[['index', 'week_int', 'week_value']]


# In[ ]:


# starttime = timeit.default_timer()
# print("The start time is :",starttime)

# extraction_settings = ComprehensiveFCParameters()

# features = extract_features(data_tseries_unpivot, column_id='index', column_sort='week_int',
#                              default_fc_parameters=extraction_settings,
#                              impute_function=impute) 

# print("Total training time is :", timeit.default_timer() - starttime)


# In[ ]:


# features.to_csv('')


# In[ ]:





# In[ ]:





# In[3]:


from dask.distributed import Client


# In[4]:


client = Client()


# In[5]:


client


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import dask.dataframe as dd


# In[7]:


df = dd.read_csv(CO_timeseries_unpivot_path)


# In[11]:


df.head()


# In[ ]:


# data_tseries_unpivot_10 = df.head(190)


# In[ ]:


# data_tseries_unpivot_10 = data_tseries_unpivot_10[['index', 'week_int', 'week_value']]


# In[ ]:


# data_tseries_unpivot_10.tail()


# In[10]:


df = df[['index', 'week_int', 'week_value']]


# In[ ]:


# each = 787 features


# In[12]:


X = extract_features(df,
                     column_id="index", column_sort="week_int",
                     pivot=False)


# In[13]:


X


# In[ ]:


X.compute()


# In[ ]:




