import os
import torch

class Trainer:
    
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, 
                 epochs, device, verbose, patience, no_stop):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.early_stopping = EarlyStopping(verbose, patience, no_stop)
        
    def fit(self, filename):
        train_losses, val_losses = [], []

        for epoch in range(1,self.epochs+1):
            train_loss = self.__train()
            evaluator = Evaluator(self.model, self.loss_fn, self.val_loader, self.device)
            val_loss = evaluator.eval()
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

    def __train(self):
        self.model.train()
        epoch_loss = 0.0
        for batch_idx, (inputs, target) in enumerate(self.train_loader):
            inputs, target = inputs.to(self.device), target.to(self.device)
            # get prediction
            output = self.model(inputs)
            loss = self.loss_fn(output, target)
            # clear previous gradients 
            self.optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # performs updates using calculated gradients
            self.optimizer.step()
            epoch_loss += loss.item()

        return  epoch_loss/len(self.train_loader)
    
    def load_checkpoint(self, filename):
        epoch, loss = 0.0, 0.0
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            name = os.path.basename(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f'=> Loaded checkpoint {name} (best epoch: {epoch}, validation error: {loss:.4f})')
        else:
            print(f'=> No checkpoint found at {filename}')

        return self.model, self.optimizer, epoch, loss
    
            
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
        
    def __init__(self, model, loss_fn, data_loader, device):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = device
        
    def eval(self):
        self.model.eval()
        epoch_loss = 0.0
        with torch.no_grad(): 
            for inputs, target in self.data_loader:
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                loss = self.loss_fn(output, target)
                epoch_loss += loss.item()

        return epoch_loss/len(self.data_loader)