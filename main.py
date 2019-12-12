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
    parser.add_argument('-i', '--iteration', type=int, default=10)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-p', '--patience', type=int, default=16)
    parser.add_argument('-w', '--workers', type=int, default=4)
    parser.add_argument('-c', '--cuda', default=0)
    parser.add_argument('-s', '--step', default=5)
    parser.add_argument('--email', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-seed', action='store_true', dest='no_seed')
    parser.add_argument('--no-stop', action='store_true', dest='no_stop')
    parser.add_argument('--small-dataset', action='store_true', dest='small_dataset')
    parser.add_argument('--convlstm', action='store_true')   
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--chirps', action='store_true')
    
    return parser.parse_args()

def log_mean_std(losses, times, iteration, util):
    losses_mean, losses_std = np.mean(losses), np.std(losses)
    times_mean, times_std = np.mean(times), np.std(times)
    times_mean_readable = util.to_readable_time(times_mean)
    print('\nErrors: ', losses)
    print('\nTimes: ', times)
    print('-----------------------')
    print(f'Mean and standard deviation after {iteration} iterations')
    print(f'=> Test Error: mean: {losses_mean:.4f}, std: {losses_std:.6f}')
    print(f'=> Training time: mean_readable: {times_mean_readable}, mean: {times_mean:.4f}, std: {times_std:.6f}')
    print('-----------------------')
    return {'test_errors_mean': losses_mean,
            'test_errors_std': losses_std,
            'train_times_mean': times_mean_readable,
            'train_times_std': times_std}
            
def run(builder, iteration, util):
    test_losses, train_times = [], []
    for i in range(iteration):                                 
        model_info = builder.run_model(i)
        test_losses.append(model_info['test_error'])
        train_times.append(model_info['train_time'])
                
    if (iteration == 1):
        return model_info
    else:
        new_model_info = log_mean_std(test_losses, train_times, iteration, util)
        new_model_info['loss_type'] = model_info['loss_type']
        new_model_info['dataset'] = model_info['dataset']
        return new_model_info


if __name__ == '__main__':    
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    message, model_descr = None, None
    if (args.convlstm): 
        model_descr = 'ConvLSTM'
    else:
        model_descr = 'STConvS2s'
    
    model_builder = MLBuilder(model_descr, args.version, args.plot, 
                               args.no_seed, args.verbose, args.small_dataset, 
                               args.no_stop, args.epoch, args.patience, device, 
                               args.workers, args.convlstm, args.mae, 
                               args.chirps, args.step)
                               
    print(f'RUN MODEL: {model_descr}')
    print(f'Device: {device}') 
    # start time is saved when creating an instance of Util
    util = Util(model_descr, version=args.version)  
    try:                                
        message = run(model_builder, args.iteration, util)
        message['step'] = args.step
        message['hostname'] = platform.node()
    except Exception as e:
        traceback.print_exc()
        message = '=> Error: ' + str(e)
    util.send_email(message, args.email)
