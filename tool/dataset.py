import h5py
import xarray as xr

import torch
from torch.utils.data import Dataset

class NetCDFDataset(Dataset):

    def __init__(self, dataset, test_split=0, validation_split=0, is_validation=False, is_test=False):
        super(NetCDFDataset, self).__init__()
        
        # orignal data format batch x time x latitude x longitude x channel
        sr = Splitter(test_split, validation_split)
        if (is_test):
            data = sr.split_test(dataset)
        elif (is_validation):
            data = sr.split_validation(dataset)
        else:
            data = sr.split_train(dataset)
        
        # data format batch x channel x time x latitude x longitude
        self.X = torch.from_numpy(data.x.values).float().permute(0, 4, 1, 2, 3)
        self.y = torch.from_numpy(data.y.values).float().permute(0, 4, 1, 2, 3)
        
        del data
        
    def __getitem__(self, index):
        return (self.X[index,:,:5,:,:], self.y[index,:,:,:,:])

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
            

class H5Dataset(Dataset):

    def __init__(self, file_path, samples, validation_split=0, isValidation=False, isTest=False):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path,'r')
        # data format batch x time x latitude x longitude x channel
        pred = h5_file.get('pred')[...]
        real = h5_file.get('real')[...]
        numpy_pred = pred[:samples]
        numpy_real = real[:samples]
        
        if (validation_split):
            split = int(numpy_pred.shape[0] * (1. - validation_split))
        
            if(isValidation):
                numpy_pred = numpy_pred[split:]
                numpy_real = numpy_real[split:]
            else:
                numpy_pred = numpy_pred[:split]
                numpy_real = numpy_real[:split]
        
        if (isTest):
            numpy_pred = pred[samples:-1]
            numpy_real = real[samples:-1]
        
        # data format batch x channel x time x latitude x longitude
        self.X = torch.from_numpy(numpy_pred).float().permute(0, 4, 1, 2, 3)
        self.y = torch.from_numpy(numpy_real).float().permute(0, 4, 1, 2, 3)
        
        del pred; del real; del numpy_pred; del numpy_real

    def __getitem__(self, index):
        return (self.X[index,:,:,:,:], self.y[index,:,:,:,:])

    def __len__(self):
        return self.X.shape[0]