

import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# import optuna

import timeit
import warnings
warnings.simplefilter(action='ignore')

import numpy as np











CO_data_path = '/home/ubuntu/workspace_rohan/project/data/processed/CO_processed_timeseries_stan.csv'
report_path = '/home/ubuntu/workspace_rohan/project/reports/model_performance/work_v5.3_standardized.txt'





# n_trials = 15





report_object = open(report_path,'a')





CO_data = pd.read_csv(CO_data_path)




CO_data['fecha_de_visita_dt'] = pd.to_datetime(CO_data['fecha_de_visita'], format="%Y-%m-%d")




not_bought_count = CO_data.bought_in_the_visit.value_counts()[0]
bought_count = CO_data.bought_in_the_visit.value_counts()[1]

baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
baseline, bought_count, not_bought_count





# print("Baseline: ", baseline, file=report_object)





features = ['week_1_standardized', 'week_2_standardized', 'week_3_standardized',
       'week_4_standardized', 'week_5_standardized', 'week_6_standardized',
       'week_7_standardized', 'week_8_standardized', 'week_9_standardized',
       'week_10_standardized', 'week_11_standardized', 'week_12_standardized',
       'week_13_standardized', 'week_14_standardized', 'week_15_standardized',
       'week_16_standardized', 'week_17_standardized', 'week_18_standardized',
       'week_19_standardized', 'cod_canal', 'cod_giro', 'cod_subgiro',
       'desc_region', 'desc_subregion', 'desc_division', 'cod_zona', 'ruta',
       'cod_modulo', 'categoria', 'marca', 'desc_sabor', 'desc_tipoenvase',
       'desc_subfamilia', 'contenido',
       'product_sales_amount_last_3m', 'product_trnx_last_3m',
       'normalized_rotation', 'normalized_freq', 'total_sales_last_3m',
       'total_trnx_last_3m', 'ratio_sales_last_3m', 'ratio_trnx_last_3m',
       'bought_last_year_flag', 'prod_coverage_bucket']










# print('\n### Direct train/test (XGBoost)')
# print('\n### Direct train/test (XGBoost)', file=report_object)





# X = CO_data[features]
# y = CO_data["bought_in_the_visit"]





# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)





# model = XGBClassifier(use_label_encoder = False, n_estimators = 500,
#         eval_metric='logloss')
# starttime = timeit.default_timer()
# print("The start time is :",starttime, file=report_object)

# model.fit(X_train, y_train)

# print("Total training time is :", timeit.default_timer() - starttime, file=report_object)




# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
# auc = roc_auc_score(y_test, y_pred_proba[:, 1])
# print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
# mtx = confusion_matrix(y_test, predictions)
# print('confusion_matrix', mtx, file=report_object)
# crep = classification_report(y_test, predictions)
# print('classification_report', crep, file=report_object)



# report_object.close()

# report_object = open(report_path,'a')



# print("\n### Direct cross validation (XGBoost)")

# print("\n### Direct cross validation (XGBoost)", file=report_object)





# X = CO_data[features]
# y = CO_data["bought_in_the_visit"]

# seed = 7

# kfold = StratifiedKFold(n_splits=5)
# model = XGBClassifier(use_label_encoder = False, n_estimators = 500,
#         eval_metric='logloss')


# starttime = timeit.default_timer()
# print("The start time is :",starttime, file=report_object)

# results = cross_val_score(model, X, y, cv=kfold)

# print("Total training time is :", timeit.default_timer() - starttime, file=report_object)

# print("Accuracy: %.2f%%" % (results.mean()*100), file=report_object)

# print("Accuracy each fold:", results, file=report_object)

# model = XGBClassifier(use_label_encoder = False, n_estimators = 500,
#         eval_metric='logloss')
# X = CO_data[features]
# y = CO_data["bought_in_the_visit"]
# seed = 7
# kfold = StratifiedKFold(n_splits=5)

# starttime = timeit.default_timer()
# print("The start time is :",starttime, file=report_object)

# results = cross_val_score(model, X, y, scoring='roc_auc',cv=kfold)

# print("Total training time is :", timeit.default_timer() - starttime, file=report_object)

# print("AUC: %.2f%%" % (results.mean()*100), file=report_object)

# print("AUC each fold:", results, file=report_object)

# report_object.close()
# report_object = open(report_path,'a')











# print("\n### Direct held out (last 2 months) (XGBoost)")
# print("\n### Direct held out (last 2 months) (XGBoost)", file=report_object)





# train = CO_data[CO_data['fecha_de_visita_dt'] < pd.to_datetime('2021-03-31', format="%Y-%m-%d")]
# test = CO_data[CO_data['fecha_de_visita_dt'] >= pd.to_datetime('2021-03-31', format="%Y-%m-%d")]

# X_train = train[features]
# X_test = test[features]
# y_train = train["bought_in_the_visit"]
# y_test = test["bought_in_the_visit"]





# model = XGBClassifier(use_label_encoder = False, n_estimators = 500,
#         eval_metric='logloss')
# starttime = timeit.default_timer()
# print("The start time is :",starttime, file=report_object)

# model.fit(X_train, y_train)

# print("Total training time is :", timeit.default_timer() - starttime, file=report_object)




# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
# auc = roc_auc_score(y_test, y_pred_proba[:, 1])
# print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
# mtx = confusion_matrix(y_test, predictions)
# print('confusion_matrix', mtx, file=report_object)
# crep = classification_report(y_test, predictions)
# print('classification_report', crep, file=report_object)




# report_object.close()

# report_object = open(report_path,'a')




print("\n### Direct train/test (Balanced Random Forest)")
print("\n### Direct train/test (Balanced Random Forest)", file=report_object)





X = CO_data[features]
y = CO_data["bought_in_the_visit"]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)



model = BalancedRandomForestClassifier(
        n_estimators=500,
        random_state=seed
        
    )

starttime = timeit.default_timer()
print("The start time is :",starttime, file=report_object)

model.fit(X_train, y_train)

print("Total training time is :", timeit.default_timer() - starttime, file=report_object)




y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
mtx = confusion_matrix(y_test, predictions)
print('confusion_matrix', mtx, file=report_object)
crep = classification_report(y_test, predictions)
print('classification_report', crep, file=report_object)



report_object.close()
report_object = open(report_path,'a')










print("\n### Direct cross validation (Balanced Random Forest)")
print("\n### Direct cross validation (Balanced Random Forest)", file=report_object)





X = CO_data[features]
y = CO_data["bought_in_the_visit"]

kfold = StratifiedKFold(n_splits=5)


model = BalancedRandomForestClassifier(
        n_estimators=500,
        random_state=seed
        
    )

starttime = timeit.default_timer()
print("The start time is :",starttime, file=report_object)

results = cross_val_score(model, X, y, cv=kfold)

print("Total training time is :", timeit.default_timer() - starttime, file=report_object)

print("Accuracy: %.2f%%" % (results.mean()*100), file=report_object)

print("Accuracy each fold:", results, file=report_object)

model = BalancedRandomForestClassifier(
        n_estimators=500,
        random_state=seed
        
    )

X = CO_data[features]
y = CO_data["bought_in_the_visit"]
seed = 7
kfold = StratifiedKFold(n_splits=5)

starttime = timeit.default_timer()
print("The start time is :",starttime, file=report_object)

results = cross_val_score(model, X, y, scoring='roc_auc',cv=kfold)

print("Total training time is :", timeit.default_timer() - starttime, file=report_object)

print("AUC: %.2f%%" % (results.mean()*100), file=report_object)

print("AUC each fold:", results, file=report_object)





report_object.close()
report_object = open(report_path,'a')


















print("\n### Direct held out (last 2 months) (Balanced Random Forest)")
print("\n### Direct held out (last 2 months) (Balanced Random Forest)", file=report_object)





train = CO_data[CO_data['fecha_de_visita_dt'] < pd.to_datetime('2021-03-31', format="%Y-%m-%d")]
test = CO_data[CO_data['fecha_de_visita_dt'] >= pd.to_datetime('2021-03-31', format="%Y-%m-%d")]

seed = 7

X_train = train[features]
X_test = test[features]
y_train = train["bought_in_the_visit"]
y_test = test["bought_in_the_visit"]

model = BalancedRandomForestClassifier(
        n_estimators=500,
        random_state=seed
        
    )

starttime = timeit.default_timer()
print("The start time is :",starttime, file=report_object)

model.fit(X_train, y_train)

print("Total training time is :", timeit.default_timer() - starttime, file=report_object)




y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
auc = roc_auc_score(y_test, y_pred_proba[:, 1])
print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
mtx = confusion_matrix(y_test, predictions)
print('confusion_matrix', mtx, file=report_object)
crep = classification_report(y_test, predictions)
print('classification_report', crep, file=report_object)


report_object.close()










