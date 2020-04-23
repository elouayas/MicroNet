"""
Model Class for Training Pipeline.
Used as the model attribute of the Trainer Class.

This class is basically nothing but instanciation.
Methods from model.py should never be called.
They are either called during instanciation or by method from a Trainer object.
"""

import math

import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
import torch

from utils.models import *
from utils.tweak import BinaryConnect
from utils.layers import LabelSmoothingCrossEntropy
from utils.optim import DelayedCosineAnnealingLR, RangerLars
from utils.decorators import timed
from utils.layers import Mish
from utils.load import load_net

import config as cfg


class Model():
    
    @timed
    def __init__(self, net):
        self.device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mode       = cfg.model['mode']
        self.net        = self._init_net(net)
        self.criterion  = self._init_criterion()
        self.optimizer  = self._init_optimizer()
        self.scheduler  = self._init_scheduler()
        self.num_params = self.get_num_params(self.net)

    def get_num_params(self, net, display_all_modules=False):
        total_num_params = 0
        for n, p in net.named_parameters():
            num_params = 1
            for s in p.shape:
                num_params *= s
            if display_all_modules: print("{}: {}".format(n, num_params))
            total_num_params += num_params
        return total_num_params

    def _init_net(self, net):
        net = load_net(cfg.dataset, net, cfg.model['quantize'])
        net = net.to(self.device)
        if self.device == 'cuda':
            net = nn.DataParallel(net)
            cudnn.benchmark = True
        return net

    def _init_criterion(self):
        if cfg.model['label_smoothing']:
            return LabelSmoothingCrossEntropy(smoothing=cfg.model['smoothing'],
                                              reduction=cfg.model['reduction'])
        else:
            return CrossEntropyLoss()

    def _init_optimizer(self):
        if self.mode == 'basic':
            optimizer = SGD(self.net.parameters(),
                            lr           = 0.1,
                            momentum     = 0.9,
                            nesterov     = True,
                            weight_decay = 5e-4)
        elif self.mode == 'boosted':
            optimizer = RangerLars(self.net.parameters())
        return optimizer

    def _init_scheduler(self):
        if self.mode == 'basic':
            scheduler = ReduceLROnPlateau(self.optimizer,
                                          mode     = 'min',
                                          factor   = 0.2,
                                          patience = 20,
                                          verbose  = True)
        elif self.mode == 'alternative':
            half_train = cfg.train['nb_epochs']//2
            scheduler = DelayedCosineAnnealingLR(self.optimizer, half_train, half_train)
        return scheduler

    def binary_connect(self, bits):
        """
        should be call only in train.py by the '_init_binary_connect' method.
        likewise, '_init_binary_connect' should never be called explicitely.
        The correct way to use this method is to modify
        the 'use_binary_connect' param of the train_config dict defined in config.py
        """
        return BinaryConnect(self.net)

    def load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['state_dict'])
