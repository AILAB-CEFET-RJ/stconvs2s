import os
import numpy as np

import torch
import torch.nn.functional as F

class Trainer:
    
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, 
                 epochs, device, util, verbose, patience, no_stop):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.util = util
        self.early_stopping = EarlyStopping(verbose, patience, no_stop)
        
    def fit(self, filename, is_chirps=False):
        train_losses, val_losses = [], []

        for epoch in range(1,self.epochs+1):
            train_loss = self.__train(is_chirps)
            evaluator = Evaluator(self.model, self.loss_fn, self.optimizer, self.val_loader, self.device, self.util)
            val_loss,_ = evaluator.eval(is_test=False, is_chirps=is_chirps)
            if (self.verbose):
                print(f'Epoch: {epoch}/{self.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.early_stopping(val_loss, self.model, self.optimizer, epoch, filename)
            if (torch.cuda.is_available()):
                torch.cuda.empty_cache()
            if self.early_stopping.isToStop:
                if (self.verbose):
                    print("=> Stopped")
                break

        return train_losses, val_losses

    def __train(self, is_chirps=False):
        self.model.train()
        epoch_loss = 0.0
        mask_land = self.util.get_mask_land().to(self.device)
        for batch_idx, (inputs, target) in enumerate(self.train_loader):
            inputs, target = inputs.to(self.device), target.to(self.device)
            # get prediction
            output = self.model(inputs)
            if is_chirps:
                output = mask_land * output
            loss = self.loss_fn(output, target)
            # clear previous gradients 
            self.optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # performs updates using calculated gradients
            self.optimizer.step()
            epoch_loss += loss.item()

        return  epoch_loss/len(self.train_loader)
            
            
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    """
    
    def __init__(self, verbose, patience, no_stop):
        self.verbose = verbose
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.isToStop = False
        self.enable_stop = not no_stop
          
    def __call__(self, val_loss, model, optimizer, epoch, filename):
        is_best = bool(val_loss < self.best_loss)
        if (is_best):
            self.best_loss = val_loss
            self.__save_checkpoint(self.best_loss, model, optimizer, epoch, filename)
            self.counter = 0
        elif (self.enable_stop):
            self.counter += 1
            if (self.verbose):
                print(f'=> Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.isToStop = True
    
    def __save_checkpoint(self, loss, model, optimizer, epoch, filename):
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'loss': loss}
        torch.save(state, filename)
        if (self.verbose):
            print ('=> Saving a new best') 
        
    
class Evaluator:
        
    def __init__(self, model, loss_fn, optimizer, data_loader, device, util=None, step=0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.util = util
        self.step = int(step)
        self.device = device
       
    def eval(self, is_test=True, is_chirps=False):
        self.model.eval()
        cumulative_rmse, cumulative_mae = 0.0, 0.0
        observation_rmse, observation_mae = [0]*self.step, [0]*self.step
        loader_size = len(self.data_loader)
        mask_land = self.util.get_mask_land().to(self.device)
        with torch.no_grad(): 
            for batch_i, (inputs, target) in enumerate(self.data_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                if is_chirps:
                    output = mask_land * output    
                rmse_loss = self.loss_fn(output, target)
                mae_loss = F.l1_loss(output, target)
                cumulative_rmse += rmse_loss.item()
                cumulative_mae += mae_loss.item()
                
                if is_test:
                    #metric per observation (lat x lon) at each time step (t) 
                    for i in range(self.step):
                        output_observation = output[:,:,i,:,:]
                        target_observation = target[:,:,i,:,:]
                        rmse_loss_obs = self.loss_fn(output_observation,target_observation)
                        mae_loss_obs = F.l1_loss(output_observation, target_observation)
                        observation_rmse[i] += rmse_loss_obs.item()
                        observation_mae[i] += mae_loss_obs.item()
        
            if is_test:             
                self.util.save_examples(inputs, target, output, self.step)
                print('>>>>>>>>> Metric per observation (lat x lon) at each time step (t)')
                print('RMSE')
                print(*np.divide(observation_rmse, batch_i+1), sep = ",")
                print('MAE')
                print(*np.divide(observation_mae, batch_i+1), sep = ",")
                print('>>>>>>>>')  
                
        return cumulative_rmse/loader_size,cumulative_mae/loader_size
        
        
    def load_checkpoint(self, filename, dataset_type=None, model=None):
        if not(os.path.isabs(filename)):
            filename = os.path.join('output', dataset_type, 'checkpoints', model.lower(), filename)  
        epoch, loss = 0.0, 0.0
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            name = os.path.basename(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f'=> Loaded checkpoint {name} (best epoch: {epoch}, validation rmse: {loss:.4f})')
        else:
            print(f'=> No checkpoint found at {filename}')

        return epoch, loss