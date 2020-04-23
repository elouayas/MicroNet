import os
import numpy as np

import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, train_config, checkpoints_path):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = train_config['patience']
        self.verbose = train_config['verbose']
        self.delta = train_config['delta']
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.checkpoints_path = checkpoints_path 

        
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter.....: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if not os.path.isdir('./checkpoints/cifar100/'):
            os.mkdir('./checkpoints/cifar100/')
        if not os.path.isdir('./checkpoints/cifar10/'):
            os.mkdir('./checkpoints/cifar10/')
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            print('Saving model ...')
        torch.save(model.state_dict(), self.checkpoints_path)
        self.val_loss_min = val_loss