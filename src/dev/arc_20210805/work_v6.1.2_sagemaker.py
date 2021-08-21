import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import optuna

import timeit
import warnings
warnings.simplefilter(action='ignore')

import argparse


if __name__ == '__main__':

    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--CO_data_path', type=str, default='/home/ubuntu/workspace_rohan/project/data/processed/CO_processed_timeseries_stan.csv')
    parser.add_argument('--model_dir', type=str, default='/home/ubuntu/workspace_rohan/project/models/work_v6')
    parser.add_argument('--n_estimators', type=int, default=50)
    args, _ = parser.parse_known_args()
    CO_data_path = args.CO_data_path
    model_dir = args.model_dir
    n_estimators = args.n_estimators

    
    
    
    
    
    CO_data = pd.read_csv(CO_data_path)
    CO_data = CO_data.head(1000)

    
    
    
    
    
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

    X = CO_data[features]
    y = CO_data["bought_in_the_visit"]

    test_size = 0.33
    seed = 7
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    
    
    
    
    
    
    
    
    model = XGBClassifier(
            use_label_encoder=False,
            n_estimators=n_estimators,
            objective="binary:logistic",
            eval_metric='logloss'
        )



    model.fit(X_train, y_train)

    
    
    
    
    
    
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    predictions = [round(value) for value in y_pred]

    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(auc)

    
    
    
    
    
    
    
    
    
    model.save_model(model_dir+"/model_sklearn.json")