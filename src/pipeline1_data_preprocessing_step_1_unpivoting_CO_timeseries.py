# script to change the format of the CO_timeseries data from pivot to unpivot as required by tsfresh

import pandas as pd
import timeit

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    # data import paths
    CO_timeseries_path = '/home/ubuntu/workspace_rohan/project/data/raw/CO_timeseries.csv'
    CO_timeseries_unpivot_path = '/home/ubuntu/workspace_rohan/project/data/processed/CO_timeseries_unpivot.csv'

    
    # data import
    data_tseries = pd.read_csv(CO_timeseries_path)
    data_tseries = data_tseries.drop(["Unnamed: 0", "week_20"],axis=1)
    data_tseries.reset_index(inplace=True)

    # data unpivot
    
    data_tseries = pd.melt(data_tseries, id_vars=['index', 'fecha_de_visita', 'codigo_de_cliente', 'codigo_de_producto'], value_vars=['week_1',
                        'week_2', 'week_3', 'week_4', 'week_5', 'week_6', 'week_7', 'week_8', 'week_9', 'week_10', 'week_11'
                            , 'week_12', 'week_13', 'week_14', 'week_15', 'week_16', 'week_17', 'week_18', 'week_19'],
                          var_name='week_number', value_name='week_value')


  
    data_tseries['week_int'] = data_tseries.week_number.str.slice(5).astype(int)

    data_tseries.sort_values(['index', 'week_int'], inplace=True)

    # data save
    data_tseries.to_csv(CO_timeseries_unpivot_path, index=False)
    
    print("Total time is :", timeit.default_timer() - starttime)
