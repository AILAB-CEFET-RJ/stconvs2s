import xarray as xr
import numpy as np

import torch
from torch.utils.data import Dataset

class NetCDFDataset(Dataset):

    def __init__(self, dataset, test_split=0, validation_split=0, is_validation=False, is_test=False, is_2d_model=False, input_channels=None, output_channels=None):
        super(NetCDFDataset, self).__init__()
        
        self.is_2d_model = is_2d_model
        # orignal data format batch x time x latitude x longitude x channel
        sr = Splitter(test_split, validation_split)
        if (is_test):
            data = sr.split_test(dataset)
        elif (is_validation):
            data = sr.split_validation(dataset)
        else:
            data = sr.split_train(dataset)
        
        # Apply input_channels slicing if specified
        if input_channels is not None:
            # Take only the specified number of channels from data.x.values
            x_data = data.x.values[:, :, :, :, :input_channels]
            self.X = torch.from_numpy(x_data).float().permute(0, 4, 1, 2, 3)
        else:
            self.X = torch.from_numpy(data.x.values).float().permute(0, 4, 1, 2, 3)
        
        # Keep first 5 timesteps (after permute: dim 2)
        self.X = self.X[:, :, :5, :, :]
        
        # Apply output_channels slicing if specified
        if output_channels is not None:
            # Take only the specified number of channels from data.y.values
            y_data = data.y.values[:, :, :, :, :output_channels]
            self.y = torch.from_numpy(y_data).float().permute(0, 4, 1, 2, 3)
        else:
            self.y = torch.from_numpy(data.y.values).float().permute(0, 4, 1, 2, 3)
        
        if self.is_2d_model:
            self.X = torch.squeeze(self.X)
            self.y = torch.squeeze(self.y)
                  
        del data
        
    def __getitem__(self, index):
        return (self.X[index,:,:,:,:], self.y[index])

    def __len__(self):
        return self.X.shape[0]
  
   
    
class Splitter():
    def __init__(self, test_rate, validation_rate):
        self.test_rate = test_rate
        self.validation_rate = validation_rate / (1. - self.test_rate)
        
    def split_test(self, dataset):
        return self.__split(dataset, self.test_rate, first_part=False)
        
    def split_validation(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=False)
        
    def split_train(self, dataset):
        data = self.__split(dataset, self.test_rate, first_part=True)
        return self.__split(data, self.validation_rate, first_part=True)
    
    def __split(self, dataset, split_rate, first_part):
        if (split_rate):
            split = int(dataset.sample.size * (1. - split_rate))
            if (first_part):
                return dataset[dict(sample=slice(0,split))]
            else:
                return dataset[dict(sample=slice(split, None))]