import sys
import platform
import pandas as pd
import numpy as np
import time
import itertools 
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm  
import statsmodels.tsa.api as smt
import argparse as arg

from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from multiprocessing import Pool 
from tool.utils import Util

import warnings
warnings.filterwarnings('ignore')
import xarray as xr

def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('--chirps', action='store_true')
    parser.add_argument('-s', '--step', default=5)
    
    return parser.parse_args()
    
def get_dataset_file(chirps):
    dataset_name, dataset_file = None, None 
    if (chirps):
        dataset_file = 'data/baseline-chirps-1981-2019.nc'
        dataset_name = 'chirps'
    else:
        dataset_file = 'data/baseline-ucar-1979-2015.nc'
        dataset_name = 'cfsr'
    
    return dataset_name, dataset_file

def create_test_sequence(dataset, n_steps_out):
    """split the dataset into samples"""
    y = [] 
    for i in range(len(dataset)):
        out_end_ix = i + n_steps_out
        if out_end_ix > len(dataset):
            break
        seq_y = dataset[i:out_end_ix]
        y.append(seq_y)
       
    return np.array(y)

def run_arima(df, chirps, step):
    series = None
    rmse_val, mae_val = 0.,0. 
    rmse_mean, mae_mean = -999., -999.
    lat = df['lat'].unique()
    lon = df['lon'].unique()
    try:
        series = df['precip'] if (chirps) else df['air_temp']
        if ((series > 0).any()):
            split = len(series) - (step + 5)
            train = series[:split].values
            test = series[split:].values
            test_sequence = create_test_sequence(test, step)
            for observation, sequence in zip(test,test_sequence):
                start_index = len(train)
                end_index = start_index + (step-1)
                model = SARIMAX(train, order=(5,0,1)) 
                results = model.fit(disp=False) 
                pred_sequence = results.predict(start=start_index, end=end_index, dynamic=False)
                rmse_val += rmse(sequence, pred_sequence) 
                mae_val += mean_absolute_error(sequence, pred_sequence)
                np.append(train,observation) 
            
            rmse_mean = rmse_val/len(test_sequence) 
            mae_mean = mae_val/len(test_sequence)
            print(f'\n=> Model ARIMA lat: {lat}, lon: {lon}')
            print(f'RMSE: {rmse_mean:.8f}')
            print(f'MAE: {mae_mean:.8f}')    
        else:
            print(f'\n** lat: {lat}, lon: {lon} has all zero values')
    except Exception as e:
        print(f'\n## lat: {lat}, lon: {lon} error: {e}')
    
    sys.stdout.flush()
    return (rmse_mean, mae_mean)


def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted) + 1e-6)


if __name__ == '__main__': 
    print('RUN MODEL: ARIMA')
    args = get_arguments()
    dataset_name, dataset_file = get_dataset_file(args.chirps)     
    ds = xr.open_mfdataset(dataset_file)
    with Pool() as pool:
        i = range(ds.lat.size)
        index_list = list(itertools.product(i,i))
        # separate time series based on each location
        ds_list = [ds.isel(lat=index[0],lon=index[1]).to_dataframe() for index in index_list]
        util = Util('ARIMA') 
        results = pool.starmap(run_arima, zip(ds_list, 
                                              itertools.repeat(args.chirps), 
                                              itertools.repeat(int(args.step))))
        results = np.array(results)
        pool.close()
        pool.join()

        print('Elapsed time', util.get_time_info()['elapsed_time'])
        rmse_list = [result[0] for result in results if result[0] >= 0]  
        mae_list = [result[1] for result in results if result[1] >= 0]
        rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
        mae_mean, mae_std = np.mean(mae_list), np.std(mae_list)
        print('\nRMSE: ', rmse_list)
        print('\nMAE: ', mae_list)
        print('-----------------------')
        print(f'Mean and standard deviation')
        print(f'=> RMSE: mean: {rmse_mean:.4f}, std: {rmse_std:.6f}')
        print(f'=> MAE: mean: {mae_mean:.4f}, std: {mae_std:.6f}')
        print('-----------------------')
        message = {'rmse_mean': rmse_mean,
                  'rmse_std': rmse_std,
                  'mae_mean': mae_mean,
                  'mae_std': mae_std,
                  'dataset_name': dataset_name,
                  'step': args.step,
                  'hostname': platform.node()}
        util.send_email(message)
        