import pandas as pd

import warnings
warnings.simplefilter(action='ignore')

CO_processed_path = '/home/ubuntu/workspace_rohan/project/data/raw/CO_processed.csv'
CO_timeseries_path = '/home/ubuntu/workspace_rohan/project/data/raw/CO_timeseries.csv'
CO_preprocessed_path = '/home/ubuntu/workspace_rohan/project/data/processed/CO_processed_timeseries_stan.csv'

data = pd.read_csv(CO_processed_path)
data = data.drop(["Unnamed: 0"],axis=1)

data_tseries = pd.read_csv(CO_timeseries_path)
data_tseries = data_tseries.drop(["Unnamed: 0"],axis=1)

data.normalized_rotation.fillna(-1, inplace=True)
data.normalized_freq.fillna(-1, inplace=True)

standardize_columns = ['week_1',
       'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8',
       'week_9', 'week_10', 'week_11', 'week_12', 'week_13', 'week_14',
       'week_15', 'week_16', 'week_17', 'week_18', 'week_19']

df = data_tseries[standardize_columns]
df = df.div(df.max(axis=1), axis=0).fillna(0)
df.columns = [str(col) + '_standardized' for col in df.columns]
data_tseries[list(df.columns)] = df

data_tseries = pd.merge(data_tseries, data, how='left', left_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'], right_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'])

data_tseries = data_tseries.dropna()

data_tseries.to_csv(CO_preprocessed_path, index=False)