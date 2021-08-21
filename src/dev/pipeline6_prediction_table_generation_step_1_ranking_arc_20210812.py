# script to generate the required precition table with ranks based on probability 

import timeit

import pandas as pd
import lightgbm as lgb
import json


if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    exp_id = 'rawweeks_'
    test_heldout_format_path = '../data/processed/model_data/test_heldout_format.csv'
    model_save_prepath = '../models/'
    prediction_table_prepath = '../data/prediction/'
    features_lable_path = '../data/feature_labels.json'
    

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features['features_current']


    print('\n### heldout format (LGB)')

    model = lgb.Booster(model_file=model_save_prepath+ exp_id +"lgb_without_optuna_heldout_format.txt")
    test = pd.read_csv(test_heldout_format_path)
    test['prediction probability'] = model.predict(test[features])
    prediction_table = test[['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto', 'prediction probability', 'bought_in_the_visit']].sort_values(['codigo_de_cliente', 'fecha_de_visita', 'prediction probability'], ascending=False)
    prediction_table['rank'] = 1
    prediction_table['rank'] = prediction_table.groupby(['codigo_de_cliente', 'fecha_de_visita'])['rank'].cumsum()
    prediction_table.to_csv(prediction_table_prepath+ exp_id +'lgb_without_optuna_heldout_format.csv', index=False)

    print("Total time is :", timeit.default_timer() - starttime)