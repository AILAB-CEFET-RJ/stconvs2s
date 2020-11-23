import warnings
warnings.filterwarnings('ignore')

import platform
import traceback
import numpy as np
import os
import argparse as arg
from tool.utils import Util
from ml_builder import MLBuilder
import torch

def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-v', '--version', default=0)
    parser.add_argument('-i', '--iteration', type=int, default=3)
    parser.add_argument('-e', '--epoch', type=int, default=80)
    parser.add_argument('-b', '--batch', type=int, default=15)
    parser.add_argument('-p', '--patience', type=int, default=16)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-c', '--cuda', default=0)
    parser.add_argument('-s', '--step', default=5)
    parser.add_argument('-m', '--model', default='stconvs2s')
    parser.add_argument('-l', '--num-layers', type=int, dest='num_layers', default=3)
    parser.add_argument('-d', '--hidden-dim', type=int, dest='hidden_dim', default=32)
    parser.add_argument('-k', '--kernel-size', type=int, dest='kernel_size', default=5)
    parser.add_argument('-t', '--pre-trained', default=None, dest='pre_trained')
    parser.add_argument('--email', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-seed', action='store_true', dest='no_seed')
    parser.add_argument('--no-stop', action='store_true', dest='no_stop')
    parser.add_argument('--small-dataset', action='store_true', dest='small_dataset')
    parser.add_argument('--chirps', action='store_true')
       
    return parser.parse_args()
    
def log_mean_std(rmse_losses, mae_losses, times, times_epochs, iteration, util):
    rmse_loss_mean, rmse_loss_std = np.mean(rmse_losses), np.std(rmse_losses)
    mae_loss_mean, mae_loss_std = np.mean(mae_losses), np.std(mae_losses)
    times_mean, times_std = np.mean(times), np.std(times)
    times_epochs_mean, times_epochs_std = np.mean(times_epochs), np.std(times_epochs)
    
    times_mean_readable = util.to_readable_time(times_mean)
    times_epochs_mean_readable = util.to_readable_time(times_epochs_mean)
    
    print('\nRMSE: ', rmse_losses)
    print('\nMAE: ', mae_losses)
    print('\nTraining times: ', times)
    print('\nTraining times/epochs: ', times_epochs)
    print('-----------------------')
    print(f'Mean and standard deviation after {iteration} iterations')
    print(f'=> Test RMSE: mean: {rmse_loss_mean:.4f}, std: {rmse_loss_std:.6f}')
    print(f'=> Test MAE: mean: {mae_loss_mean:.4f}, std: {mae_loss_std:.6f}')
    print(f'=> Training times: mean_readable: {times_mean_readable}, mean: {times_mean:.4f}, std: {times_std:.6f}')
    
    if times_epochs_mean > 0.:
        print('=> Training times/epochs: mean_readable: '\
              f'{times_epochs_mean_readable}, mean: {times_epochs_mean:.4f}, std: {times_epochs_std:.6f}')
    print('-----------------------')
    return {'test_rmse_mean': rmse_loss_mean,
            'test_rmse_std': rmse_loss_std,
            'test_mae_mean': mae_loss_mean,
            'test_mae_std': mae_loss_std,
            'train_times_mean': times_mean_readable,
            'train_times_std': times_std}
            
def run(builder, iteration, util):
    test_rmse, test_mae, train_times, train_times_epochs = [],[],[],[]
    for i in range(iteration):                                 
        model_info = builder.run_model(i)
        if (iteration == 1):
            return model_info
        test_rmse.append(model_info['test_rmse'])
        test_mae.append(model_info['test_mae'])
        train_times.append(model_info['train_time'])
        train_times_epochs.append(model_info['train_time_epochs'])
                
    new_model_info = log_mean_std(test_rmse, test_mae, train_times, train_times_epochs, iteration, util)
    new_model_info['dataset'] = model_info['dataset']
    return new_model_info


if __name__ == '__main__':    
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
    device = torch.device('cpu')
    device_descr = 'CPU'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_descr = 'GPU'
        
    message = None
    model_builder = MLBuilder(args, device)       
    print(f'RUN MODEL: {args.model.upper()}')
    print(f'Device: {device_descr}') 
    print(f'Settings: {args}')
    # start time is saved when creating an instance of Util
    util = Util(args.model, version=args.version)  
    try:                                
        message = run(model_builder, args.iteration, util)
        message['step'] = args.step
        message['hostname'] = platform.node()
    except Exception as e:
        traceback.print_exc()
        message = '=> Error: ' + str(e)
    util.send_email(message, args.email)
