
import timeit

import pandas as pd
import lightgbm as lgb


import json

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    exp_id = '20210822v2_lgb_with_optuna_on_cv_on_auc_on_full_data_normalized_weeks_without_tsfresh_features_and_categorical_original_saved_model_on_heldout_train_format'
    
    test_path = '../data/processed/test_heldout_format.csv'
    model_save_prepath = '../models/'
    features_lable_path = '../data/feature_labels.json'
    prediction_table_prepath = '../data/prediction/'
    df_recom_path = '../data/db_dump/20210805_get_recom_qty_dump.csv'
    df_sales_path = '../data/db_dump/20210805_get_venta_neta_dump.csv'

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features['features_current']

    
    df_recom = pd.read_csv(df_recom_path)
    df_recom.recomm_qt.fillna(0, inplace=True)
    df_recom['recomm_qt'] = df_recom['recomm_qt'].astype(int)
    df_recom['fecha_de_visita'] = df_recom.fecha_de_visita.astype(str)
    df_recom['codigo_de_cliente'] = df_recom.codigo_de_cliente.astype(str)

    df_sales = pd.read_csv(df_sales_path)

    df_sales.fecha_de_visita = df_sales.fecha_de_visita.astype(str)
    df_sales.codigo_de_cliente = df_sales.codigo_de_cliente.astype(str)
    df_sales.codigo_de_producto = df_sales.codigo_de_producto.astype(str)
   

    
    ############################################################
    
    model = lgb.Booster(model_file=model_save_prepath+ exp_id +".txt")
    test = pd.read_csv(test_path)

    y_pred = model.predict(test[features])

    predictions = [round(value) for value in y_pred]
 
    ## Trimmed test
    test['prediction probability'] = y_pred
    test['predictions'] = predictions
    prediction_table = test[['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto', 'prediction probability', 'predictions', 'bought_in_the_visit']].sort_values(['codigo_de_cliente', 'fecha_de_visita', 'prediction probability'], ascending=False)
    prediction_table['rank'] = 1
    prediction_table['rank'] = prediction_table.groupby(['codigo_de_cliente', 'fecha_de_visita'])['rank'].cumsum()
    
      
    prediction_table['fecha_de_visita'] = prediction_table.fecha_de_visita.astype(str)
    prediction_table['codigo_de_cliente'] = prediction_table.codigo_de_cliente.astype(str)
    prediction_table['codigo_de_producto'] = prediction_table.codigo_de_producto.astype(str)
    prediction_table = pd.merge(prediction_table, df_recom, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita'], right_on=['codigo_de_cliente', 'fecha_de_visita'])
    
    prediction_table.to_csv(prediction_table_prepath+ exp_id +'_prediction.csv', index=False)
    
    prediction_table = prediction_table[prediction_table['rank'] <= prediction_table['recomm_qt']]
 
    prediction_table.to_csv(prediction_table_prepath+ exp_id +'_prediction_trimmed.csv', index=False)
    
    print("Total time is :", timeit.default_timer() - starttime)