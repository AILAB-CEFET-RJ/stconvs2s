import numpy as np
import random as rd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

from model.convlstm import STConvLSTM
from model.stconvs2s import STConvS2S
from tool.train_evaluate import Trainer, Evaluator
from tool.dataset import NetCDFDataset
from tool.loss import RMSELoss
from tool.utils import Util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim


class MLBuilder:

    def __init__(self, model_descr, version, plot, no_seed, verbose, 
               small_dataset, no_stop, epoch, patience, device, 
               workers, convlstm, mae, chirps, step):
        
        self.model_descr = model_descr
        self.version = version
        self.plot = plot
        self.no_seed = no_seed
        self.verbose = verbose
        self.small_dataset = small_dataset
        self.no_stop = no_stop
        self.epochs = epoch
        self.patience = patience
        self.device = device
        self.dataset_type = 'small-dataset' if (self.small_dataset) else 'full-dataset'
        self.workers = workers
        self.convlstm = convlstm
        self.mae = mae
        self.chirps = chirps
        self.step = str(step) 
        
    def run_model(self, seed_number):
        self.__define_seed(seed_number)
        # Hyperparameters
        batch_size = 50
        validation_split = 0.2
        test_split = 0.2
        dataset_file = None
        # Loading the dataset
        dataset_name, dataset_file, dropout_rate = self.__get_dataset_file()
        ds = xr.open_mfdataset(dataset_file)
        if (self.small_dataset):
            ds = ds[dict(sample=slice(0,500))]

        train_dataset = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split)
        val_dataset = NetCDFDataset(ds, test_split=test_split, 
                                    validation_split=validation_split, is_validation=True)
        test_dataset = NetCDFDataset(ds, test_split=test_split, 
                                     validation_split=validation_split, is_test=True)
        if (self.verbose):
            print('[X_train] Shape:', train_dataset.X.shape)
            print('[y_train] Shape:', train_dataset.y.shape)
            print('[X_val] Shape:', val_dataset.X.shape)
            print('[y_val] Shape:', val_dataset.y.shape)
            print('[X_test] Shape:', test_dataset.X.shape)
            print('[y_test] Shape:', test_dataset.y.shape)
            print(f'Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples')

        upsample = False
        filename_prefix = dataset_name
        if (int(self.step) == 15):
            upsample = True
            filename_prefix += '_step' + self.step
            
        params = {'batch_size': batch_size, 
                  'num_workers': self.workers, 
                  'worker_init_fn': self.__init_seed}

        train_loader = DataLoader(dataset=train_dataset, shuffle=True,**params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False,**params)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params)

        # Creating the model
        model, criterion  = None, None
        if (self.convlstm):
            model = STConvLSTM(input_size=train_dataset.X.shape[3], dropout_rate=dropout_rate, upsample=upsample)
        else:
            model = STConvS2S(channels=train_dataset.X.shape[1], dropout_rate=dropout_rate, upsample=upsample)
                              
        model.to(self.device)
        
        if (self.mae):
            criterion = nn.L1Loss()
        else:
            criterion = RMSELoss()
        
        opt_params = {'lr': 0.001, 
                      'alpha': 0.9, 
                      'eps': 1e-6}
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        model_info = self.__execute_learning(model, criterion, optimizer, train_loader, val_loader, 
                                              test_loader, dataset_name, filename_prefix, dropout_rate)

        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        #util.send_email(model_info, self.email)
        return model_info


    def __execute_learning(self, model, criterion, optimizer, train_loader, val_loader, test_loader, 
                            dataset_name, filename_prefix, dropout_rate):
        criterion_name = type(criterion).__name__
        filename_prefix += '_' + criterion_name
        util = Util(self.model_descr, self.dataset_type, self.version, filename_prefix)
    
        # Training the model
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, self.epochs, 
                          self.device, self.verbose, self.patience, self.no_stop)
    
        start_timestamp = tm.time()
        train_losses, val_losses = trainer.fit(checkpoint_filename)
        end_timestamp = tm.time()
        # Error analysis
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 
                  'Epochs', 'Error', 'Error analysis', self.plot)
    
        # Load model with minimal loss after training phase
        model,_, best_epoch, val_loss = trainer.load_checkpoint(checkpoint_filename)
    
        # Evaluating the model
        evaluator = Evaluator(model, criterion, test_loader, self.device)
        test_loss = evaluator.eval()
        train_time = end_timestamp - start_timestamp
        print(f'Training time: {util.to_readable_time(train_time)}\n{self.model_descr} {criterion_name}: {test_loss:.4f}\n')
        
        return {'best_epoch': best_epoch,
                'val_error': val_loss,
                'test_error': test_loss,
                'train_time': train_time,
                'loss_type': criterion_name,
                'dropout_rate': dropout_rate,
                'dataset': dataset_name}
          
    def __define_seed(self, number):      
        if (~self.no_seed):
            # define a different seed in every iteration 
            number *= 1000
            np.random.seed(number)
            rd.seed(number)
            torch.manual_seed(number)
            torch.cuda.manual_seed(number)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        np.random.seed(number*1000)
        
    def __get_dataset_file(self):
        dataset_file, dataset_name = None, None
        dropout_rate = 0.
        if (self.chirps):
            dataset_file = 'data/dataset-chirps-1981-2019-seq5-ystep' + self.step + '.nc'
            dataset_name = 'chirps'
            dropout_rate = 0.8 if (self.convlstm) else 0.2
            
        else:
            dataset_file = 'data/dataset-ucar-1979-2015-seq5-ystep' + self.step + '.nc'
            dataset_name = 'cfsr'
        
        return dataset_name, dataset_file, dropout_rate
    
