# script to evaluate the performance of the trained models

import timeit

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (precision_score, recall_score, accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

from sklearn.model_selection import KFold
import json
import numpy as np

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    exp_id = '20210822v1_lgb_without_optuna_normalized_weeks_without_tsfresh_features_and_categorical_original_saved_model_on_heldout_train_format'
    
    test_path = '../data/processed/test_heldout_format.csv'
    CO_timeseries_featurized_path = '../data/processed/full_data.csv'
    model_save_prepath = '../models/'
    models_evaluation_report_path = '../reports/' + exp_id + '.txt'
    features_lable_path = '../data/feature_labels.json'
#     prediction_table_prepath = '../data/prediction/'
    df_recom_path = '../data/db_dump/20210805_get_recom_qty_dump.csv'
    df_sales_path = '../data/db_dump/20210805_get_venta_neta_dump.csv'
    precision_report_prepath = '../reports/full_precision_reports/'
    
    seed = 7

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
    
    report_object = open(models_evaluation_report_path,'a')
    kfold = KFold(n_splits=5)

    print('\n### Performance on Test Dataset', file=report_object)
    
    

    
    ############################################################
    
    model = lgb.Booster(model_file=model_save_prepath+ exp_id +".txt")
    test = pd.read_csv(test_path)

    not_bought_count = test.bought_in_the_visit.value_counts()[0]
    bought_count = test.bought_in_the_visit.value_counts()[1]
    baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
    print('Test Baseline:', baseline, file=report_object)


    y_test = test['bought_in_the_visit']
    y_pred = model.predict(test[features])

    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0), file=report_object)
    recall = recall_score(y_test, predictions)
    print("Recall: %.2f%%" % (recall * 100.0), file=report_object)
    precision = precision_score(y_test, predictions)
    print("Precision: %.2f%%" % (precision * 100.0), file=report_object)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC: %.2f%%" % (auc * 100.0), file=report_object)
    mtx = confusion_matrix(y_test, predictions)
    print('confusion_matrix', mtx, file=report_object)
    crep = classification_report(y_test, predictions)
    print('classification_report', crep, file=report_object)
    
    
    ## Trimmed test
    print('\n### Metrics on Trimmed test data', file=report_object)
    test['prediction probability'] = y_pred
    test['predictions'] = predictions
    prediction_table = test[['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto', 'prediction probability', 'predictions', 'bought_in_the_visit']].sort_values(['codigo_de_cliente', 'fecha_de_visita', 'prediction probability'], ascending=False)
    prediction_table['rank'] = 1
    prediction_table['rank'] = prediction_table.groupby(['codigo_de_cliente', 'fecha_de_visita'])['rank'].cumsum()
#     prediction_table.to_csv(prediction_table_prepath+ exp_id +'.csv', index=False)
      
    prediction_table['fecha_de_visita'] = prediction_table.fecha_de_visita.astype(str)
    prediction_table['codigo_de_cliente'] = prediction_table.codigo_de_cliente.astype(str)
    prediction_table['codigo_de_producto'] = prediction_table.codigo_de_producto.astype(str)
    prediction_table = pd.merge(prediction_table, df_recom, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita'], right_on=['codigo_de_cliente', 'fecha_de_visita'])
    prediction_table = prediction_table[prediction_table['rank'] <= prediction_table['recomm_qt']]
    prediction_table = pd.merge(prediction_table, df_sales, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto'], right_on=['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto'])

    prediction_table['venta_neta_dolar'] = prediction_table['venta_neta_dolar'].fillna(0.00)
    prediction_table['executed_flag'] = prediction_table['venta_neta_dolar'].apply(lambda x: 1 if x > 0.00 else 0)
    
#     precision = precision_score(prediction_table['bought_in_the_visit'], prediction_table['predictions'])
#     print("Precision: %.2f%%" % (precision * 100.0), file=report_object)
    
    temp = prediction_table[prediction_table['predictions']==1]
    precision = round(temp[temp['bought_in_the_visit']==1].shape[0]/temp.shape[0],4)
    print("\nPrecision: for class 1 %.2f%%" % (precision * 100.0), file=report_object)
    
    temp = prediction_table[prediction_table['predictions']==0]
    precision = round(temp[temp['bought_in_the_visit']==0].shape[0]/temp.shape[0],4)
    print("\nPrecision: for class 0 %.2f%%" % (precision * 100.0), file=report_object)
    
    
    client_level_precision = prediction_table.groupby(['codigo_de_cliente']).agg({'codigo_de_producto': 'count', 'executed_flag': 'sum'}).reset_index()
    client_level_precision['precision'] = client_level_precision['executed_flag']/client_level_precision['codigo_de_producto']

    client_and_visit_level_precision = prediction_table.groupby(['fecha_de_visita', 'codigo_de_cliente']).agg({'codigo_de_producto': 'count', 'executed_flag': 'sum'}).reset_index()
    client_and_visit_level_precision['non_executed_flag'] = client_and_visit_level_precision['codigo_de_producto']-client_and_visit_level_precision['executed_flag']
    client_and_visit_level_precision['precision_execution_flag'] = client_and_visit_level_precision['executed_flag']/client_and_visit_level_precision['codigo_de_producto']
    client_and_visit_level_precision['precision_non_execution_flag'] = client_and_visit_level_precision['non_executed_flag']/client_and_visit_level_precision['codigo_de_producto']
    

    precision_class_1 = client_and_visit_level_precision['precision_execution_flag'].mean()
    precision_class_2 = client_and_visit_level_precision['precision_non_execution_flag'].mean()
    
    client_level_precision.to_csv(precision_report_prepath+ exp_id + '_client_level_precision.csv', index=False)
    client_and_visit_level_precision.to_csv(precision_report_prepath+ exp_id + '_client_and_visit_level_precision.csv', index=False)
    print('\nClient and Visit Level Mean precision for class 1:', round(precision_class_1,2), file=report_object)
    print('\nClient and Visit Level Mean precision for class 0:', round(precision_class_2,2), file=report_object)
    
#     CV Evaluation
    print('\n### Cross Validation on Full Dataset', file=report_object)
    full = pd.read_csv(CO_timeseries_featurized_path)
    

    not_bought_count = full.bought_in_the_visit.value_counts()[0]
    bought_count = full.bought_in_the_visit.value_counts()[1]
    baseline = round(max(bought_count, not_bought_count)/ (bought_count + not_bought_count),2)
    print('Full Dataset Baseline:', baseline, file=report_object)
    
    
    val_scores = []
    acc_scores = []
    rec_scores = []
    pre_scores = []
    auc_scores = []
    overall_precision_after_trimming_scores = []
    for train_idx, val_idx in kfold.split(full):

        val_all = full.iloc[val_idx]
        y_val = full.iloc[val_idx]['bought_in_the_visit']
        val = full.iloc[val_idx][features]

        y_train = full.iloc[train_idx]['bought_in_the_visit']
        train = full.iloc[train_idx][features]

        train.columns = list(range(0, len(list(train.columns))))
        train = lgb.Dataset(train, y_train)
        model = lgb.train(params={'learning_rate':0.05}, train_set=train)

        y_pred = model.predict(val)
        predictions = [round(value) for value in y_pred]
        
        acc_scores.append(round(accuracy_score(y_val, predictions),2))
        rec_scores.append(round(recall_score(y_val, predictions),2))
        pre_scores.append(round(precision_score(y_val, predictions),2))
        auc_scores.append(round(roc_auc_score(y_val, y_pred),2))
        
        val_all['prediction probability'] = y_pred
        val_all['predictions'] = predictions
        prediction_table = val_all[['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto', 'prediction probability', 'predictions', 'bought_in_the_visit']].sort_values(['codigo_de_cliente', 'fecha_de_visita', 'prediction probability'], ascending=False)
        prediction_table['rank'] = 1
        prediction_table['rank'] = prediction_table.groupby(['codigo_de_cliente', 'fecha_de_visita'])['rank'].cumsum()
        prediction_table['fecha_de_visita'] = prediction_table.fecha_de_visita.astype(str)
        prediction_table['codigo_de_cliente'] = prediction_table.codigo_de_cliente.astype(str)
        prediction_table['codigo_de_producto'] = prediction_table.codigo_de_producto.astype(str)
        prediction_table = pd.merge(prediction_table, df_recom, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita'], right_on=['codigo_de_cliente', 'fecha_de_visita'])
        prediction_table = prediction_table[prediction_table['rank'] <= prediction_table['recomm_qt']]
        prediction_table = pd.merge(prediction_table, df_sales, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto'], right_on=['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto'])

        prediction_table['venta_neta_dolar'] = prediction_table['venta_neta_dolar'].fillna(0.00)
        prediction_table['executed_flag'] = prediction_table['venta_neta_dolar'].apply(lambda x: 1 if x > 0.00 else 0)

        overall_precision_after_trimming = precision_score(prediction_table['bought_in_the_visit'], prediction_table['predictions'])

        overall_precision_after_trimming_scores.append(round(overall_precision_after_trimming,2))
 
    
    print("CV accuracy each fold:", acc_scores, file=report_object)
    print("CV accuracy Mean:", np.mean(np.array(acc_scores)), file=report_object)
    
    print("CV recall each fold:", rec_scores, file=report_object)
    print("CV recall Mean:", np.mean(np.array(rec_scores)), file=report_object)
    
    print("CV precision each fold:", pre_scores, file=report_object)
    print("CV precision Mean:", np.mean(np.array(pre_scores)), file=report_object)
    
    print("CV auc each fold:", auc_scores, file=report_object)
    print("CV auc Mean:", np.mean(np.array(auc_scores)), file=report_object)

    print("CV overall_precision_after_trimming each fold:", overall_precision_after_trimming_scores, file=report_object)
    print("CV overall_precision_after_trimming Mean:", np.mean(np.array(overall_precision_after_trimming_scores)), file=report_object)

    print("Evaluation report saved at -", models_evaluation_report_path)
    
    report_object.close()
    
    print("Total time is :", timeit.default_timer() - starttime)