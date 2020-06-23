"""
Model Class for Training Pipeline.
Used as the model attribute of the Trainer Class.
"""

import torch
import torch.nn as nn
from utils import DenseNet
from utils.layers import LabelSmoothingCrossEntropy
from torch.optim import SGD
from utils.optim import *
from torch.optim.lr_scheduler import (ReduceLROnPlateau, MultiStepLR,
                                      CosineAnnealingLR, CosineAnnealingWarmRestarts)
from utils.schedulers import *
from utils.decorators import timed, saveWrapper
import config as cfg


class Model():
    
    @timed
    def __init__(self, net_name):
        self.net        = DenseNet.from_name(net_name)
        self.optimizer  = self.init_optimizer(cfg.optim)
        self.scheduler  = self.init_scheduler(self.optimizer, cfg.scheduler,
                                         cfg.dataloader['train_batch_size'], cfg.train['nb_epochs'])
        self.criterion  = nn.CrossEntropyLoss()
        self.num_params = self.get_num_params(self.net)

    def get_num_params(self, net, display_all_modules=False):
        total_num_params = 0
        for n, p in net.named_parameters():
            num_params = 1
            for s in p.shape:
                num_params *= s
            if display_all_modules: print(f"{n}: {num_params}")
            total_num_params += num_params
        return total_num_params

    def init_optimizer(self, net, config):
        if config['type'] == 'SGD':
            optimizer = SGD(net.parameters(),
                            lr           = config['params']['lr'],
                            momentum     = config['params']['momentum'],
                            nesterov     = config['params']['nesterov'],
                            weight_decay = config['params']['weight_decay'])
        else:
            optimizer = Ralamb(net.parameters(),
                            lr           = config['params']['lr'],
                            betas        = config['params']['betas'],
                            eps          = config['params']['eps'],
                            weight_decay = config['params']['weight_decay'])
        if config['use_lookahead']:
            return Lookahead(optimizer,
                            alpha = config['lookahead']['alpha'],
                            k     = config['lookahead']['k'])
        else:
            return optimizer

    def init_scheduler(self, optimizer, config, batch_size, epochs):
        if config['type'] == 'ROP':
            scheduler = ReduceLROnPlateau(optimizer,
                                        mode     = config['params']['mode'],
                                        factor   = config['params']['factor'],
                                        patience = config['params']['patience'],
                                        verbose  = config['params']['verbose'])
        elif config['type'] == 'MultiStep':
            scheduler = MultiStepLR(optimizer,
                                    milestones = config['params']['milestones'],
                                    gamma      = config['params']['gamma'],
                                    last_epoch = config['params']['last_epoch'])
        elif config['type'] == 'Cosine':
            scheduler = CosineAnnealingLR(optimizer, config['params']['epochs'])
        elif config['type'] == 'WarmRestartsCosine':
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                    T_0        = config['params']['T_0'],
                                    T_mult     = config['params']['T_mult'],
                                    eta_min    = config['params']['eta_min'],
                                    last_epoch = config['params']['last_epoch'])
        else:
            scheduler = WarmupCosineLR(batch_size, 50000, epochs,
                                    config['params']['base_lr'], config['params']['target_lr'],
                                    config['params']['warm_up_epoch'], config['params']['cur_epoch'])
        if config['use_delay']:
            return DelayerScheduler(optimizer, config['delay']['delay_epochs'], scheduler)
        else:
            return scheduler

    def init_criterion(self, config):
        if config['label_smoothing']:
            return LabelSmoothingCrossEntropy(smoothing=config['smoothing'],
                                              reduction=config['reduction'])
        else:
            return nn.CrossEntropyLoss()

    



