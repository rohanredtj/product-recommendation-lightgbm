# script to divide the preprocessed data into training and testing dataset in batch manner in two formats - train/ test split format, helout format

import timeit
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    CO_timeseries_featurized_path = ("../data/processed/CO_timeseries_featurized.csv")
    CO_timeseries_model_data_prepath = ("../data/processed/model_data/")


    df_CO_timeseries_featurized = pd.read_csv(CO_timeseries_featurized_path)
    print("File import done")

    df_CO_timeseries_featurized["fecha_de_visita_dt"] = pd.to_datetime(df_CO_timeseries_featurized["fecha_de_visita"], format="%Y-%m-%d")
    print("fecha_de_visita_dt timeseries done")


    th = round(df_CO_timeseries_featurized.shape[0])
    step = 4000

    # split format
    seed = 7
    test_size = 0.1
    start = 0
    while (start < th):

        end = start+step
        print(start, end)
        df_temp = df_CO_timeseries_featurized[(df_CO_timeseries_featurized.index>=start) & (df_CO_timeseries_featurized.index<end)]
        y_temp = df_temp["bought_in_the_visit"]
        train_temp, test_temp, y_train_temp, y_test_temp = train_test_split(
        df_temp, y_temp, test_size=test_size, random_state=seed
    )
        if start==0:
            train_temp.to_csv(CO_timeseries_model_data_prepath+'train_split_format.csv', mode='a', index=False)
            test_temp.to_csv(CO_timeseries_model_data_prepath+'test_split_format.csv', mode='a', index=False)
        else:
            train_temp.to_csv(CO_timeseries_model_data_prepath+'train_split_format.csv', mode='a', header=False, index=False)
            test_temp.to_csv(CO_timeseries_model_data_prepath+'test_split_format.csv', mode='a', header=False, index=False)
        start = start+step
    print('split format done')


#     # heldout format
#     start = 0
#     while (start < th):

#         end = start+step
#         print(start, end)
#         df_temp = df_CO_timeseries_featurized[(df_CO_timeseries_featurized.index>=start) & (df_CO_timeseries_featurized.index<end)]
#         train_temp = df_temp[df_temp["fecha_de_visita_dt"] < pd.to_datetime("2021-03-31", format="%Y-%m-%d")]
#         test_temp = df_temp[df_temp["fecha_de_visita_dt"] >= pd.to_datetime("2021-03-31", format="%Y-%m-%d")]

#         if start==0:
#             train_temp.to_csv(CO_timeseries_model_data_prepath+'train_heldout_format.csv', mode='a', index=False)
#             test_temp.to_csv(CO_timeseries_model_data_prepath+'test_heldout_format.csv', mode='a', index=False)
#         else:
#             train_temp.to_csv(CO_timeseries_model_data_prepath+'train_heldout_format.csv', mode='a', header=False, index=False)
#             test_temp.to_csv(CO_timeseries_model_data_prepath+'test_heldout_format.csv', mode='a', header=False, index=False)
#         start = start+step
#     print("heldout format done")

    print("Total time is :", timeit.default_timer() - starttime)