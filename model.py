""" Base Model Class: A Lighning Module
    This class implements all the logic code.
    This model class will be the one to be fit by a Trainer
 """

import numpy as np
import torch
from pytorch_lightning.core.lightning import LightningModule
# from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Accuracy
from utils.data import get_train_dataloader, get_val_dataloader
from utils.model import DenseNet
from utils.model.init import init_criterion, init_optimizer, init_scheduler
from utils.model.layers import Cutmix
from utils.decorators import val_verbose, train_verbose

from time import time


class LightningModel(LightningModule):
    """
        LightningModule handling everything training related.
        Pytorch Lightning will be referred as Lightning in all the following.
        Many attributes and method used aren't explicitely defined here
        but comes from the LightningModule class.
        This behavior should be specify in a docstring.
        Please refer to the Lightning documentation for further details.

        Note that Lighning handles tensorboard logging, early stopping, and auto checkpoints
        for this class.
    """

    def __init__(self, config):
        """
            Config is a dataclass with 7 attributes.
            Each of this attributes is itself a dataclass.
            Please refer to config.py to see what they contain.
            All this params are saved alltogether with the weights.
            Hence one can load an already trained model and acces all its hyperparameters.
            This call to save_hyperparameters() is handled by Lightning.
            It makes something like self.hparams.dataloader.train_batch_size callable.
        """
        super().__init__()
        self.config      = config
        self.net         = DenseNet.from_name(self.config.dataset, self.config.model.net)
        self.criterion   = init_criterion(self.config.model)
        self.metric      = Accuracy()
        self.save_hyperparameters()

    def train_dataloader(self):
        return get_train_dataloader(self.config.dataset, self.config.dataloader)

    def val_dataloader(self):
        return get_val_dataloader(self.config.dataset, self.config.dataloader)

    def configure_optimizers(self):
        optimizer = init_optimizer(self.net, self.config.optimizer)
        scheduler = init_scheduler(optimizer, self.config.scheduler)
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self): # this name is explicitely needed to overload parent's method
        tqdm_dict = super().get_progress_bar_dict()
        if 'v_num' in tqdm_dict:
            del tqdm_dict['v_num']
        return tqdm_dict

    def forward(self, x):
        return self.net(x)

    def cutmix(self, inputs, targets):
        """ mix two inputs in a Cutout way """
        lam, inputs, target_a, target_b = Cutmix(inputs, targets, self.config.train.cutmix_beta)
        outputs, _ = self(inputs)
        loss_a, loss_b  = self.criterion(outputs, target_a), self.criterion(outputs, target_b)
        loss = loss_a * lam + loss_b * (1. - lam)
        return outputs, loss

    def infere(self, inputs, targets):
        """ infere on the giving inputs and compute the loss using targets
            returns loss, layers (layers is for GKD)
        """
        proba = np.random.rand(1)
        if self.config.train.use_cutmix and proba < self.config.train.cutmix_p:
            outputs, loss = self.cutmix(inputs, targets)
        else:
            outputs, layers = self(inputs) # second return value is layers for GKD
            loss = self.criterion(outputs, targets)
        acc  = self.metric(outputs, targets)
        return loss, acc, layers

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        loss, acc, _ = self.infere(inputs, targets) # third return values is layers for GKD
        log = {'acc': acc}
        return {'loss': loss, 'logs': log, 'progress_bar': log}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, _ = self(inputs)
        val_loss = self.criterion(outputs, targets)
        #val_acc  = self.metric(outputs, targets) # this line doubles validation time WTF ???
        #return {'val_loss': val_loss, 'val_acc': val_acc}
        return {'val_loss': val_loss}

    """
    @train_verbose
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc  = torch.stack([x['acc']  for x in outputs]).mean()
        lr       = self.trainer.optimizers[0].param_groups[0]['lr']
        logs = {'loss': avg_loss, 'train_acc': avg_acc, 'lr': lr}
        return {'loss': avg_loss, 'log': logs}
    """

    # @val_verbose
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #avg_acc  = torch.stack([x['val_acc']  for x in outputs]).mean()
        #logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        #return {'val_loss': avg_loss, 'log': logs}
        return {'val_loss': avg_loss}
