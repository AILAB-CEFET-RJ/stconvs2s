import os

def get_param_value(param_name, type_operation):
    value = None
    default_values = {
        'dataset': 'cfsr',
        'step':'5',
        'dropout_rate': 0.,
        'lr': 0.001,
        'model': 'stconvs2s',
        'version': 1
    }    
    try:
        value = type_operation(os.environ[param_name])
    except Exception as e:
        value = default_values[param_name]
    return value


if __name__ == '__main__': 
    dataset  = get_param_value('dataset', str)
    step  = get_param_value('step', str)
    lr = get_param_value('lr', float)
    model = get_param_value('model', str)
    version = get_param_value('version', int)
    dropout_rate = get_param_value('dropout_rate', float)
    param = None

    if dataset == 'chirps':
        dataset_file = '../data/dataset-chirps-1981-2019-seq5-ystep' + step + '.nc'
        input_size = 50
        if dropout_rate == 0.:
            dropout_rate = 0.8 if (model == 'convlstm') else 0.2

    elif dataset == 'cfsr':
        dataset_file = '../data/dataset-ucar-1979-2015-seq5-ystep' + step + '.nc'
        input_size = 32
            
    if model == 'stconvs2s':
        param_stconvs2s = {
            1: {'encoder_layer_size': 2, 'decoder_layer_size': 2, 'kernel_size': 3, 'filter_size': 64},
            2: {'encoder_layer_size': 3, 'decoder_layer_size': 3, 'kernel_size': 3, 'filter_size': 32},
            3: {'encoder_layer_size': 3, 'decoder_layer_size': 3, 'kernel_size': 3, 'filter_size': 64},
            4: {'encoder_layer_size': 3, 'decoder_layer_size': 3, 'kernel_size': 5, 'filter_size': 32}
        }
        param = param_stconvs2s[version]
        
    elif model == 'convlstm':
        param_convlstm = {
            1: {'layer_size': 2, 'kernel_size': 3, 'hidden_dim': 64},
            2: {'layer_size': 3, 'kernel_size': 3, 'hidden_dim': 32},
            3: {'layer_size': 3, 'kernel_size': 3, 'hidden_dim': 64},
            4: {'layer_size': 3, 'kernel_size': 5, 'hidden_dim': 32}
        }
        param = param_convlstm[version]
    
    print(f'version = {version}')
    print(f'dataset = {dataset_file}')
    print(f'input_size = {input_size}')
    print(f'step = {step}')
    print(f'dropout rate = {dropout_rate}')
    print(f'learning rate = {lr}')
    print(f'param = {param}')