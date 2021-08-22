# script to batch preprocess the data

import timeit
import pandas as pd
import json

import warnings
warnings.simplefilter(action='ignore')

from tsfresh import extract_features
from dask.distributed import Client
from tsfresh.utilities.distribution import ClusterDaskDistributor


if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
#     client = Client()
#     ip = client.scheduler.address.split('//')[-1]
    
    features_lable_path = '../data/feature_labels.json'
    CO_timeseries_featurized_prepath = '../data/processed/'
    CO_timeseries_path = '../data/raw/CO_timeseries.csv'
    CO_timeseries_unpivot_path = '../data/processed/CO_timeseries_unpivot.csv'

    CO_timeseries_unpivot_shared_prepath = '../data/processed/CO_timeseries_unpivot_shared/'

    CO_timeseries_unpivot_featurized_shared_prepath = '../data/processed/CO_timeseries_unpivot_featurized_shared/'

    CO_processed_path = '../data/raw/CO_processed.csv'

    df_pd = pd.read_csv(CO_timeseries_unpivot_path)
    df_pd = df_pd[['index', 'week_int', 'week_value']]
    
    with open(features_lable_path, "rb") as fh:
        features = json.load(fh)
    columns_save = features['columns_save']
    
    th = round(df_pd.shape[0]/19)
    step = 4000
    
    # make the batches of the unpivot data
    
#     start = 0
#     while (start < th):
#         end = start+step
#         print(start, end)
#         df_temp = df_pd[(df_pd['index']>=start) & (df_pd['index']<end)]
#         df_temp.to_csv(CO_timeseries_unpivot_shared_prepath+'CO_timeseries_unpivot_shared_'+str(start)+'_'+str(end)+'.csv', index=False)
#         start = start+step

#     # apply tsfresh in batches
#     start = 0
#     while (start < th):
#         end = start+step
#         print(start, end)
#         df_temp = pd.read_csv(CO_timeseries_unpivot_shared_prepath+'CO_timeseries_unpivot_shared_'+str(start)+'_'+str(end)+'.csv')
#         Distributor = ClusterDaskDistributor(address=ip)
#         X = extract_features(timeseries_container=df_temp,
#                              column_id='index', column_sort='week_int',
#                              disable_progressbar=False,
#                              distributor=Distributor)

#         X.to_csv(CO_timeseries_unpivot_featurized_shared_prepath+'CO_timeseries_unpivot_featurized_shared_'+str(start)+'_'+str(end)+'.csv', index=False)
#         start = start+step

    # join tsfresh featurized data with filling na etc and joining with CO_processed data


    data = pd.read_csv(CO_processed_path)
    data = data.drop(["Unnamed: 0"],axis=1)
    data.normalized_rotation.fillna(-1, inplace=True)
    data.normalized_freq.fillna(-1, inplace=True)
    na_check = data.columns

    standardize_columns = ['week_1',
       'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8',
       'week_9', 'week_10', 'week_11', 'week_12', 'week_13', 'week_14',
       'week_15', 'week_16', 'week_17', 'week_18', 'week_19']

    df_CO_timeseries_raw = pd.read_csv(CO_timeseries_path)
    df_CO_timeseries_raw.drop(['Unnamed: 0'], axis=1, inplace=True)
    df_CO_timeseries_featurized = pd.DataFrame()
    start = 0
    while (start < th):

        end = start+step
        print(start, end)
        df_temp = df_CO_timeseries_raw.iloc[start:end, :]
        df_temp.reset_index(inplace=True)

        df_feat = pd.read_csv(CO_timeseries_unpivot_featurized_shared_prepath+'CO_timeseries_unpivot_featurized_shared_'+str(start)+'_'+str(end)+'.csv')    
        df_temp_full = pd.concat([df_temp, df_feat], axis=1)

        df_temp_new = df_temp_full[standardize_columns]
        df_temp_new = df_temp_new.div(df_temp_new.max(axis=1), axis=0).fillna(0)
        df_temp_new.columns = [str(col) + '_standardized' for col in df_temp_new.columns]
        df_temp_full[list(df_temp_new.columns)] = df_temp_new

        df_temp_full = pd.merge(df_temp_full, data, how='left', left_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'], right_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'])
        df_temp_full = df_temp_full.dropna(subset=na_check)
        df_temp_full.fillna(0, inplace=True)
        df_temp_full = df_temp_full[columns_save]
        if start==0:
            df_temp_full.to_csv(CO_timeseries_featurized_prepath+'CO_timeseries_featurized.csv', mode='a', index=False)
        else:
            df_temp_full.to_csv(CO_timeseries_featurized_prepath+'CO_timeseries_featurized.csv', mode='a', header=False, index=False)
        start = start+step
    
    print("Total time is :", timeit.default_timer() - starttime)
    
    



