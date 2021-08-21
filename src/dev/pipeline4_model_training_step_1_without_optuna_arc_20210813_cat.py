# script to train the models and save the trained models

import timeit
import pandas as pd
import json
import lightgbm as lgb
import re

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    train_heldout_format_path = '../data/processed/model_data/train_heldout_format.csv'
    model_save_prepath = '../models/'
    features_lable_path = '../data/feature_labels.json'
    exp_id = '20210813v1_lgb_without_optuna_heldout_format_normalized_weeks_without_tsfresh_features_and_categorical_original'

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features['features_current']

    seed = 7
    print('\n### split heldout format (LGB)')

    train = pd.read_csv(train_heldout_format_path)
    print('Training Data Loaded')
    y_train = train["bought_in_the_visit"]
    train = train[features]
    
    categorical_features = ['cod_canal', 'cod_giro', 'cod_subgiro', 'desc_region', 'desc_subregion',
                       'desc_division', 'cod_zona', 'ruta', 'cod_modulo', 'categoria', 'marca', 
                       'desc_sabor', 'desc_tipoenvase', 'desc_subfamilia', 'contenido', 'prod_coverage_bucket']
    
    categorical_features_index = []
    i = 0
    for col in train.columns:
        if col in categorical_features:
            categorical_features_index.append(i)
        i = i + 1
    
    train.columns = list(range(0, len(list(train.columns))))
    train = lgb.Dataset(train, y_train)
    print('Model Train Start')
    model = lgb.train(params={'learning_rate':0.05}, categorical_feature=categorical_features_index,train_set=train)
    model.save_model(model_save_prepath+ exp_id +".txt", num_iteration=model.best_iteration)
        
    print("Total time is :", timeit.default_timer() - starttime)