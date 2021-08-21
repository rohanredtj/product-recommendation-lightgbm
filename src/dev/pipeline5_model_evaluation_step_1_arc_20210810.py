# script to evaluate the performance of the trained models

import timeit

import pandas as pd
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import json

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    test_split_format_path = '../data/processed/model_data/test_split_format.csv'
    test_heldout_format_path = '../data/processed/model_data/test_heldout_format.csv'
    CO_timeseries_featurized_path = '../data/processed/CO_timeseries_featurized.csv'
    model_save_prepath = '../models/'
    models_evaluation_report_path = '../reports/models_evaluation_report.txt'
    features_lable_path = '../data/feature_labels.json'

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features['features_current']

    report_object = open(models_evaluation_report_path,'a')
    kfold = StratifiedKFold(n_splits=5)
    nrows = 500000
    print('nrows', nrows, file=report_object)

    ############################################################
    print('\n### split train/test format (XGBoost)')
    print('\n### split train/test format (XGBoost)', file=report_object)

    model = XGBClassifier()
    model.load_model(model_save_prepath+"xgboost_without_optuna_split_format.json")
    test = pd.read_csv(test_split_format_path)

    not_bought_count = test.bought_in_the_visit.value_counts()[0]
    bought_count = test.bought_in_the_visit.value_counts()[1]
    baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
    print('Test Baseline:', baseline, file=report_object)


    y_test = test['bought_in_the_visit']
    test = test[features]

    y_pred = model.predict(test)
    y_pred_proba = model.predict_proba(test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
    mtx = confusion_matrix(y_test, predictions)
    print('confusion_matrix', mtx, file=report_object)
    crep = classification_report(y_test, predictions)
    print('classification_report', crep, file=report_object)

    ## CV Evaluation
#     test = pd.read_csv(CO_timeseries_featurized_path, nrows=nrows)

#     not_bought_count = test.bought_in_the_visit.value_counts()[0]
#     bought_count = test.bought_in_the_visit.value_counts()[1]
#     baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
#     print('All Data Baseline:', baseline, file=report_object)

#     y_test = test['bought_in_the_visit']
#     test = test[features]

#     results = cross_val_score(model, test, y_test, cv=kfold)
#     print("CV Mean Accuracy: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV Accuracy each fold:", results, file=report_object)
#     results = cross_val_score(model, test, y_test, scoring='roc_auc', cv=kfold)
#     print("CV Mean AUC: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV AUC each fold:", results, file=report_object)

    ############################################################
    print('\n### split train/test format (Balanced Random Forest)')
    print('\n### split train/test format (Balanced Random Forest)', file=report_object)

    test = pd.read_csv(test_split_format_path)
    y_test = test['bought_in_the_visit']
    test = test[features]

    with open(model_save_prepath+"balanced_rf_without_optuna_split_format.pickle", 'rb') as handle:
        model = pickle.load(handle)

    y_pred = model.predict(test)
    y_pred_proba = model.predict_proba(test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
    mtx = confusion_matrix(y_test, predictions)
    print('confusion_matrix', mtx, file=report_object)
    crep = classification_report(y_test, predictions)
    print('classification_report', crep, file=report_object)

    ## CV Evaluation

#     test = pd.read_csv(CO_timeseries_featurized_path, nrows=nrows)
#     y_test = test['bought_in_the_visit']
#     test = test[features]
#     results = cross_val_score(model, test, y_test, cv=kfold)

#     print("CV Mean Accuracy: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV Accuracy each fold:", results, file=report_object)
#     results = cross_val_score(model, test, y_test, scoring='roc_auc', cv=kfold)
#     print("CV Mean AUC: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV AUC each fold:", results, file=report_object)

    ############################################################
    print('\n### heldout format (XGBoost)')
    print('\n### heldout format (XGBoost)', file=report_object)

    model = XGBClassifier()
    model.load_model(model_save_prepath+"xgboost_without_optuna_heldout_format.json")
    test = pd.read_csv(test_heldout_format_path)

    not_bought_count = test.bought_in_the_visit.value_counts()[0]
    bought_count = test.bought_in_the_visit.value_counts()[1]
    baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
    print('Test Baseline:', baseline, file=report_object)

    y_test = test['bought_in_the_visit']
    test = test[features]

    y_pred = model.predict(test)
    y_pred_proba = model.predict_proba(test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
    mtx = confusion_matrix(y_test, predictions)
    print('confusion_matrix', mtx, file=report_object)
    crep = classification_report(y_test, predictions)
    print('classification_report', crep, file=report_object)

    ## CV Evaluation
#     test = pd.read_csv(CO_timeseries_featurized_path, nrows=nrows)
#     y_test = test['bought_in_the_visit']
#     test = test[features]

#     results = cross_val_score(model, test, y_test, cv=kfold)
#     print("CV Mean Accuracy: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV Accuracy each fold:", results, file=report_object)
#     results = cross_val_score(model, test, y_test, scoring='roc_auc', cv=kfold)
#     print("CV Mean AUC: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV AUC each fold:", results, file=report_object)

    ############################################################
    print('\n### heldout format (Balanced Random Forest)')
    print('\n### heldout format (Balanced Random Forest)', file=report_object)

    test = pd.read_csv(test_heldout_format_path)
    y_test = test['bought_in_the_visit']
    test = test[features]

    with open(model_save_prepath+"balanced_rf_without_optuna_heldout_format.pickle", 'rb') as handle:
        model = pickle.load(handle)

    y_pred = model.predict(test)
    y_pred_proba = model.predict_proba(test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
    mtx = confusion_matrix(y_test, predictions)
    print('confusion_matrix', mtx, file=report_object)
    crep = classification_report(y_test, predictions)
    print('classification_report', crep, file=report_object)

    ## CV Evaluation
#     test = pd.read_csv(CO_timeseries_featurized_path, nrows=nrows)
#     y_test = test['bought_in_the_visit']
#     test = test[features]
#     results = cross_val_score(model, test, y_test, cv=kfold)
#     print("CV Mean Accuracy: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV Accuracy each fold:", results, file=report_object)
#     results = cross_val_score(model, test, y_test, scoring='roc_auc', cv=kfold)
#     print("CV Mean AUC: %.2f%%" % (results.mean()*100), file=report_object)
#     print("CV AUC each fold:", results, file=report_object)

    ############################################################
    print("Evaluation report saved at -", models_evaluation_report_path)
    
    print("Total time is :", timeit.default_timer() - starttime)