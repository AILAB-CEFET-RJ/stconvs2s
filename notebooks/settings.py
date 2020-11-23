import os

from model.stconvs2s import STConvS2S_R, STConvS2S_C
from model.baselines import *

def get_param_value(param_name, type_operation):
    value = None
    default_values = {
        'dataset': 'cfsr',
        'step':'5',
        'dropout_rate': 0.,
        'lr': 0.001,
        'model_name': 'stconvs2s',
        'version': 1,
        'cuda': '0',
        'only_training': False
    }    
    try:
        value = type_operation(os.environ[param_name])
    except Exception as e:
        value = default_values[param_name]
    print(f'{param_name} = {value}')
    
    return value


if __name__ == '__main__': 
    dataset  = get_param_value('dataset', str)
    step  = get_param_value('step', str)
    lr = get_param_value('lr', float)
    model_name = get_param_value('model_name', str)
    version = get_param_value('version', int)
    cuda = get_param_value('cuda', str)
    dropout_rate = get_param_value('dropout_rate', float)
    only_training = get_param_value('only_training', bool)
    epochs = 50
    batch_size = 15
    validation_split = 0.2
    test_split = 0.2
    small_dataset = False
    is_chirps = True if dataset == 'chirps' else False
    
    models = {
        'stconvs2s-r': STConvS2S_R,
        'stconvs2s-c': STConvS2S_C,
        'convlstm': STConvLSTM,
        'predrnn': PredRNN,
        'mim': MIM
    }
    dropout_rates = {
        'predrnn': 0.5,
        'mim': 0.5
    }
    datasets = {
        'cfsr': {
            'dataset_file':'../data/dataset-ucar-1979-2015-seq5-ystep' + step + '.nc',
            'models_param': {
                1: {'num_layers': 2, 'kernel_size': 3, 'hidden_dim': 64},
                2: {'num_layers': 3, 'kernel_size': 3, 'hidden_dim': 32},
                3: {'num_layers': 3, 'kernel_size': 3, 'hidden_dim': 64},
                4: {'num_layers': 3, 'kernel_size': 5, 'hidden_dim': 32}
            }   
        },
        'chirps': {
            'dataset_file':'../data/dataset-chirps-1981-2019-seq5-ystep' + step + '.nc',
            'models_param': {
                1: {'num_layers': 1, 'kernel_size': 3, 'hidden_dim': 16},
                2: {'num_layers': 2, 'kernel_size': 3, 'hidden_dim': 8},
                3: {'num_layers': 2, 'kernel_size': 3, 'hidden_dim': 16},
                4: {'num_layers': 2, 'kernel_size': 5, 'hidden_dim': 8}
            } 
        }
    } 
    dataset_file = datasets[dataset]['dataset_file']
    model_param = datasets[dataset]['models_param'][version]
    # not executing dropout search
    if not(only_training) and model_name in dropout_rates:
        dropout_rate = dropout_rates[model_name] 
    
    print(f'epochs = {epochs}')
    print(f'batch_size = {batch_size}')
    print(f'validation_split = {validation_split}')
    print(f'test_split = {test_split}')
    print(f'dataset_file = {dataset_file}')
    print(f'model_param = {model_param}')