import pandas as pd
import timeit
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    CO_timeseries_path = '../data/raw/CO_timeseries.csv' # Or prediction raw data path with no label column
    CO_processed_path = '../data/raw/CO_processed.csv' # And corresponding prediction raw data path
    CO_timeseries_model_data_prepath = ("../data/processed/")
    
    seed = 7
    test_size = 0.1 
    heldout_threshold = "2021-03-31"
    
    standardize_columns = ['week_1',
       'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8',
       'week_9', 'week_10', 'week_11', 'week_12', 'week_13', 'week_14',
       'week_15', 'week_16', 'week_17', 'week_18', 'week_19']
    
    df_CO_timeseries_raw = pd.read_csv(CO_timeseries_path)
    df_CO_timeseries_raw.drop(['Unnamed: 0'], axis=1, inplace=True)

    data = pd.read_csv(CO_processed_path)
    data = data.drop(["Unnamed: 0"],axis=1)
    data.normalized_rotation.fillna(-1, inplace=True)
    data.normalized_freq.fillna(-1, inplace=True)
    
    df_temp_new = df_CO_timeseries_raw[standardize_columns]
    df_temp_new = df_temp_new.div(df_temp_new.max(axis=1), axis=0).fillna(0)
    df_temp_new.columns = [str(col) + '_standardized' for col in df_temp_new.columns]
    df_CO_timeseries_raw[list(df_temp_new.columns)] = df_temp_new

    data = pd.merge(df_CO_timeseries_raw, data, how='left', left_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'], right_on=['fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'])
    data = data.dropna()
    data.fillna(0, inplace=True)
    
    data["fecha_de_visita_dt"] = pd.to_datetime(data["fecha_de_visita"], format="%Y-%m-%d")
    
    print("Saving Full Data")
    data.to_csv(CO_timeseries_model_data_prepath+'full_data.csv', index=False)
    
    if 'bought_in_the_visit' not in data.columns:
        print("Total time is :", timeit.default_timer() - starttime)
        exit()
    
    # split format
    y_data = data["bought_in_the_visit"]
    train, test, y_train, y_test = train_test_split(
    data, y_data, test_size=test_size, random_state=seed
    )
    
    print("Saving Train Split Format")
    train.to_csv(CO_timeseries_model_data_prepath+'train_split_format.csv', index=False)
    print("Saving Test Split Format")
    test.to_csv(CO_timeseries_model_data_prepath+'test_split_format.csv', index=False)
    
    # heldout format
    train = data[data["fecha_de_visita_dt"] < pd.to_datetime(heldout_threshold, format="%Y-%m-%d")]
    test = data[data["fecha_de_visita_dt"] >= pd.to_datetime(heldout_threshold, format="%Y-%m-%d")]
    
    print("Saving Train Heldout Format")
    train.to_csv(CO_timeseries_model_data_prepath+'train_heldout_format.csv', index=False)
    print("Saving Test Heldout Format")
    test.to_csv(CO_timeseries_model_data_prepath+'test_heldout_format.csv', index=False)
      
    
    print("Total time is :", timeit.default_timer() - starttime)
    
    