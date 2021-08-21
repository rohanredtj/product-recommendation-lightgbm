#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import timeit
import warnings
warnings.simplefilter(action='ignore')


# In[4]:


import optuna


# In[5]:


CO_processed_path = '/home/ubuntu/workspace_rohan/project/data/raw/CO_processed.csv'
CO_timeseries_path = '/home/ubuntu/workspace_rohan/project/data/raw/CO_timeseries.csv'


# In[6]:


data = pd.read_csv(CO_processed_path)
data = data.drop(["Unnamed: 0"],axis=1)


# In[7]:


data_tseries = pd.read_csv(CO_timeseries_path)
data_tseries = data_tseries.drop(["Unnamed: 0"],axis=1)


# In[8]:


data_tseries = pd.merge(data_tseries, data, how='left', left_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'], right_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'])


# In[9]:


data_tseries = data_tseries.dropna()


# In[ ]:





# In[10]:


not_bought_count = data_tseries.bought_in_the_visit.value_counts()[0]
bought_count = data_tseries.bought_in_the_visit.value_counts()[1]

baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
baseline, bought_count, not_bought_count


# In[ ]:





# In[11]:


features = ['week_1',
       'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8',
       'week_9', 'week_10', 'week_11', 'week_12', 'week_13', 'week_14',
       'week_15', 'week_16', 'week_17', 'week_18', 'week_19', 'cod_canal', 'cod_giro', 'cod_subgiro',
       'desc_region', 'desc_subregion', 'desc_division', 'cod_zona', 'ruta',
       'cod_modulo', 'categoria', 'marca', 'desc_sabor', 'desc_tipoenvase',
       'desc_subfamilia', 'contenido',
       'product_sales_amount_last_3m', 'product_trnx_last_3m',
       'normalized_rotation', 'normalized_freq', 'total_sales_last_3m',
       'total_trnx_last_3m', 'ratio_sales_last_3m', 'ratio_trnx_last_3m',
       'bought_last_year_flag', 'prod_coverage_bucket']


# In[12]:


X = data_tseries[features]
y = data_tseries["bought_in_the_visit"]


# In[15]:


seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[16]:


def objective(trial):
    # Fit train data

    learning_rate = trial.suggest_loguniform("lr", 1e-3, 1)
    n_estimators = trial.suggest_int("n_est", 1, 750)
    #alp = trial.suggest_categorical('alp', [1.0, 1.3, 1.4])
    max_depth = trial.suggest_int("m_depth", 1, 20)
    reg_alpha = trial.suggest_loguniform("alpha", 1e-10, 1)
    reg_lamb = trial.suggest_loguniform("reg_lamb", 1e-10, 1)
    gamma = trial.suggest_float("gamma", 0.1, 1)
    col_tree = trial.suggest_float("colt_tree", 0.5, 1)
    min_child = trial.suggest_loguniform("min_child", 0.1, 20)
    subsamp = trial.suggest_float("subsamp", 0.01, 1)

    model = XGBClassifier(
        use_label_encoder=False,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=min_child,
        gamma=gamma,
        subsample=subsamp,
        colsample_bytree=col_tree,
        objective="binary:logistic",
        seed=seed,
        reg_alpha = reg_alpha,
        max_depth = max_depth,
        reg_lambda = reg_lamb
    )

    model.fit(X_train, y_train)

    print("Model completed")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    predictions = [round(value) for value in y_pred]
    
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    return auc


# In[17]:


starttime = timeit.default_timer()
print("The start time is :",starttime)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)

print("Total training time is :", timeit.default_timer() - starttime)


# In[18]:


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




