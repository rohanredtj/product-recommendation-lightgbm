# script to cut the prediction table based on required ranks and calulated the precision score 

import timeit

import pandas as pd

if __name__ == '__main__':
    
    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    
    exp_id = 'rawweeks_'
    prediction_table_prepath = '../data/prediction/'
    mean_precision_report_path = '../reports/' + exp_id + 'mean_precision_report.txt'
    report_prepath = '../reports/'
    df_recom_path = '../data/db_dump/20210805_get_recom_qty_dump.csv'
    df_sales_path = '../data/db_dump/20210805_get_venta_neta_dump.csv'

    report_object = open(mean_precision_report_path,'a')

    df_recom = pd.read_csv(df_recom_path)
    df_recom.recomm_qt.fillna(0, inplace=True)
    df_recom['recomm_qt'] = df_recom['recomm_qt'].astype(int)
    df_recom['fecha_de_visita'] = df_recom.fecha_de_visita.astype(str)
    df_recom['codigo_de_cliente'] = df_recom.codigo_de_cliente.astype(str)

    df_sales = pd.read_csv(df_sales_path)

    df_sales.fecha_de_visita = df_sales.fecha_de_visita.astype(str)
    df_sales.codigo_de_cliente = df_sales.codigo_de_cliente.astype(str)
    df_sales.codigo_de_producto = df_sales.codigo_de_producto.astype(str)

    def cut_and_compute(prediction_table):

        prediction_table['fecha_de_visita'] = prediction_table.fecha_de_visita.astype(str)
        prediction_table['codigo_de_cliente'] = prediction_table.codigo_de_cliente.astype(str)
        prediction_table['codigo_de_producto'] = prediction_table.codigo_de_producto.astype(str)


        prediction_table = pd.merge(prediction_table, df_recom, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita'], right_on=['codigo_de_cliente', 'fecha_de_visita'])



        prediction_table = prediction_table[prediction_table['rank'] <= prediction_table['recomm_qt']]



        prediction_table = pd.merge(prediction_table, df_sales, how='left', left_on=['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto'], right_on=['codigo_de_cliente', 'fecha_de_visita', 'codigo_de_producto'])

        prediction_table['venta_neta_dolar'] = prediction_table['venta_neta_dolar'].fillna(0.00)
        prediction_table['executed_flag'] = prediction_table['venta_neta_dolar'].apply(lambda x: 1 if x > 0.00 else 0)


        client_level_precision = prediction_table.groupby(['codigo_de_cliente']).agg({'codigo_de_producto': 'count', 'executed_flag': 'sum'}).reset_index()
        client_level_precision['precision'] = client_level_precision['executed_flag']/client_level_precision['codigo_de_producto']

        client_and_visit_level_precision = prediction_table.groupby(['fecha_de_visita', 'codigo_de_cliente']).agg({'codigo_de_producto': 'count', 'executed_flag': 'sum'}).reset_index()
        client_and_visit_level_precision['precision'] = client_and_visit_level_precision['executed_flag']/client_and_visit_level_precision['codigo_de_producto']

        precision = client_and_visit_level_precision['precision'].mean()

        return client_level_precision, client_and_visit_level_precision, precision


    exp_id = ''
    
    print('\n### heldout format (LGB)')
    print('\n### heldout format (LGB)', file=report_object)
    prediction_table = pd.read_csv(prediction_table_prepath+ exp_id +'lgb_without_optuna_heldout_format.csv')
    client_level_precision, client_and_visit_level_precision, precision = cut_and_compute(prediction_table)
    client_level_precision.to_csv(report_prepath+ exp_id +'client_level_precision_report_lgb_without_optuna_heldout_format.csv', index=False)
    client_and_visit_level_precision.to_csv(report_prepath+ exp_id +'client_and_visit_level_precision_report_lgb_without_optuna_heldout_format.csv', index=False)
    print('\nMean precision', precision)
    print('\nMean precision', precision, file=report_object)

    report_object.close()