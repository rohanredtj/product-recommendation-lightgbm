import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score, precision_score
import lightgbm as lgb
import json
from sklearn.model_selection import KFold
import numpy as np
import timeit

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    model_save_prepath = '../models/'
    features_lable_path = '../data/feature_labels.json'
    CO_timeseries_featurized_path = '../data/processed/full_data.csv'
    df_recom_path = '../data/db_dump/20210805_get_recom_qty_dump.csv'
    df_sales_path = '../data/db_dump/20210805_get_venta_neta_dump.csv'
    train_path = '../data/processed/train_heldout_format.csv'
    exp_id = '20210822v3_lgb_with_optuna_on_cv_on_overall_precision_after_trimming_on_full_data_normalized_weeks_without_tsfresh_features_and_categorical_original_saved_model_on_heldout_train_format'

    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    features = features['features_current']

    n_trials = 30
    n_splits = 5
    
    kfold = KFold(n_splits=n_splits)
    
    df_recom = pd.read_csv(df_recom_path)
    df_recom.recomm_qt.fillna(0, inplace=True)
    df_recom['recomm_qt'] = df_recom['recomm_qt'].astype(int)
    df_recom['fecha_de_visita'] = df_recom.fecha_de_visita.astype(str)
    df_recom['codigo_de_cliente'] = df_recom.codigo_de_cliente.astype(str)

    df_sales = pd.read_csv(df_sales_path)

    df_sales.fecha_de_visita = df_sales.fecha_de_visita.astype(str)
    df_sales.codigo_de_cliente = df_sales.codigo_de_cliente.astype(str)
    df_sales.codigo_de_producto = df_sales.codigo_de_producto.astype(str)

    
    full = pd.read_csv(CO_timeseries_featurized_path)
    
    def objective(trial):
        
        params = {
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-3, 1),
            'n_estimators' : trial.suggest_int("n_estimators", 1, 750),
            'max_depth' : trial.suggest_int("max_depth", 1, 20),
            'reg_alpha' : trial.suggest_loguniform("reg_alpha", 1e-10, 1),
            'num_leaves' : trial.suggest_int("num_leaves", 2, 100)
        }

#         # CV on AUC
#         val_scores = [] 
#         for train_idx, val_idx in kfold.split(full):
#             y_val = full.iloc[val_idx]['bought_in_the_visit']
#             val = full.iloc[val_idx][features]
#             y_train = full.iloc[train_idx]['bought_in_the_visit']
#             train = full.iloc[train_idx][features]
#             train.columns = list(range(0, len(list(train.columns))))
#             train = lgb.Dataset(train, y_train)
#             model = lgb.train(params=params, train_set=train)
#             y_pred = model.predict(val)
#             auc = roc_auc_score(y_val, y_pred)
#             val_scores.append(auc)
#         auc = np.mean(np.array(val_scores))
#         return auc
    
        # CV on Overall precision after trimming
        val_scores = []
        for train_idx, val_idx in kfold.split(full):
            
            val_all = full.iloc[val_idx]
            y_val = full.iloc[val_idx]['bought_in_the_visit']
            val = full.iloc[val_idx][features]

            y_train = full.iloc[train_idx]['bought_in_the_visit']
            train = full.iloc[train_idx][features]

            train.columns = list(range(0, len(list(train.columns))))
            train = lgb.Dataset(train, y_train)
            model = lgb.train(params=params, train_set=train)

            y_pred = model.predict(val)
            predictions = [0 if value <= 0.5 else 1 for value in y_pred]
            
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
            
            val_scores.append(overall_precision_after_trimming)

        overall_precision_after_trimming = np.mean(np.array(val_scores))
        return overall_precision_after_trimming

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    train = pd.read_csv(train_path)
    y_train = train['bought_in_the_visit']
    train = train[features]

    train.columns = list(range(0, len(list(train.columns))))
    train = lgb.Dataset(train, y_train)
    model = lgb.train(params=study.best_trial.params, train_set=train)
    
    model.save_model(model_save_prepath+ exp_id +".txt", num_iteration=model.best_iteration)
    
    print("Total time is :", timeit.default_timer() - starttime)
