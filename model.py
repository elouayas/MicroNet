import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from utils.data import get_train_dataloader, get_val_dataloader
from utils.model import DenseNet
from utils.model.init import *
from utils.model.layers import Cutmix
from utils import Distillator


class Model(LightningModule):
    """
        LightningModule handling everything training related.
        Pytorch Lighning will be referred as Lighning in all the following. 
        Many attributes and method used here aren't explicitely defined here
        but comes from the LightningModule class.
        This behavior should be specify in a docstring.
        Please refer to the Lightning documentation for further details.

        Note that Lighning handles tensorboard logging, early stopping, and auto checkpoints
        for this class.
    """

    def __init__(self, config):
        """
            config contains all the hyperparameters defined in config.py
            Contains the following dicts:
                * config.dataset
                * config.dataloader 
                * config.model
                * config.optim
                * config.scheduler
                * config.train
                * config.teacher
            Please refer to config.py to see what each of this dict contains.
            All this params are saved alltogether with the weights 
            Hence one can load an already trained model and acces all its hyperparameters.
        """
        super().__init__()
        self.config      = config
        self.net         = DenseNet.from_name(config.dataset, config.model['net'])
        self.criterion   = init_criterion(self.config.model)
        self.distillator = self._init_distillation()
        self.save_hyperparameters({'dataset': config.dataset, 'dataloader': config.dataloader,
                                   'train': config.train, 'model': config.model,
                                   'optim': config.optim, 'scheduler': config.scheduler})

    def _init_distillation(self):
        """ returns another instance of the Model class,
            with an already trained net used as a teacher.
            The distillation params are accessibles via self.config.teacher
            The loading (call to .load_from_checkpoint()) is handled by Lightning) 
        """
        if not self.config.train['distillation']: 
            return None
        teacher = self(self.config.teacher['net']) 
        teacher.load_from_checkpoint(self.config.teacher['teacher_path'])
        return Distillator(self.config.dataset, teacher, self.config.teacher)

    def train_dataloader(self):
        return get_train_dataloader(self.config.dataset, self.config.dataloader)

    def val_dataloader(self):
        return get_val_dataloader(self.config.dataset, self.config.dataloader) 

    def configure_optimizers(self): 
        optimizer = init_optimizer(self.net, self.config.optim)
        scheduler = init_scheduler(optimizer, self.config.scheduler)
        return [optimizer], [scheduler]

    def forward(self, x): 
        return self.net(x)

    def cutmix(self, inputs, targets):
        lam, inputs, target_a, target_b = Cutmix(inputs, targets, self.config.train['cutmix_beta'])
        outputs, layers = self(inputs)
        loss_a, loss_b  = self.criterion(outputs, target_a), self.criterion(outputs, target_b)
        return loss_a * lam + loss_b * (1. - lam)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        r = np.random.rand(1)
        if self.config.train['use_cutmix'] and r < self.config.train['cutmix_p']:
            loss = self.cutmix(inputs, targets)
        else:
            outputs, layers = self(inputs) 
        if self.config.train['distillation']:
            loss = self.distillator.run(inputs, outputs, targets, layers)
        else:
            loss = self.criterion(outputs, targets)
        logs = {'loss': loss}
        return {'loss': loss, 'logs': logs} 

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, layers = self(inputs) 
        loss = self.criterion(outputs, targets)
        logs = {'val_loss': loss}
        return {'val_loss': loss, 'logs': logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}