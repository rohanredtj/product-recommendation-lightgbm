# script to train the models and save the trained models

import timeit
import pandas as pd
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle
import json

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    train_split_format_path = '../data/processed/model_data/train_split_format.csv'
    train_heldout_format_path = '../data/processed/model_data/train_heldout_format.csv'
    model_save_prepath = '../models/'
    features_lable_path = '../data/feature_labels.json'

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features['features_current']

    seed = 7
    nrows = 1000000
    print('nrows', nrows)
    print('\n### split train/test format (XGBoost)')

    train = pd.read_csv(train_split_format_path, nrows=nrows)
    y_train = train["bought_in_the_visit"]
    train = train[features]
    model = XGBClassifier(use_label_encoder = False,
            eval_metric='logloss',
            seed=seed)
    model.fit(train, y_train)
    model.save_model(model_save_prepath+"xgboost_without_optuna_split_format.json")


    print('\n### split train/test format (Balanced Random Forest)')

    train = pd.read_csv(train_split_format_path, nrows=nrows)
    y_train = train["bought_in_the_visit"]
    train = train[features]
    model = BalancedRandomForestClassifier(
            random_state=seed, n_jobs=-1)
    model.fit(train, y_train)
    with open(model_save_prepath+"balanced_rf_without_optuna_split_format.pickle", 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print('\n### heldout format (XGBoost)')

    train = pd.read_csv(train_heldout_format_path, nrows=nrows)
    y_train = train["bought_in_the_visit"]
    train = train[features]
    model = XGBClassifier(use_label_encoder = False,
            eval_metric='logloss',
            seed=seed)
    model.fit(train, y_train)
    model.save_model(model_save_prepath+"xgboost_without_optuna_heldout_format.json")


    print('\n### heldout format (Balanced Random Forest)')

    train = pd.read_csv(train_heldout_format_path, nrows=nrows)
    y_train = train["bought_in_the_visit"]
    train = train[features]
    model = BalancedRandomForestClassifier(
            random_state=seed, n_jobs=-1)
    model.fit(train, y_train)
    with open(model_save_prepath+"balanced_rf_without_optuna_heldout_format.pickle", 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Total time is :", timeit.default_timer() - starttime)