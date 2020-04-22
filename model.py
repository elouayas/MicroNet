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

from models import *

from utils.binary_connect import BC
from utils.label_smoothing import LabelSmoothingCrossEntropy
from utils.optim.delayed_lr_scheduler import DelayedCosineAnnealingLR
from utils.optim.ranger_lars import RangerLars
from utils.decorators import timed
from utils.mish import Mish

import config as cfg


class Model():
    
    @timed
    def __init__(self):
        self.device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mode       = cfg.model['mode']
        self.summary    = {} # will contain dataset, net, num_params, optimizer, scheduler
        self.net        = self._init_net()
        self.criterion  = self._init_criterion()
        self.optimizer  = self._init_optimizer()
        self.scheduler  = self._init_scheduler()

    def _load_net(self):
        num_classes = 100 if cfg.dataset=='cifar100' else 10
        if cfg.model['quantize']:
            if cfg.model['net']=='wide_resnet_28_10':
                net = WideResNet_28_10_Quantized(num_classes=num_classes)
            else:
                net = ResNet_Quantized(cfg.model['net'], num_classes=num_classes)
        else:
            if cfg.model['net']=='wide_resnet_28_10':
                net = WideResNet_28_10(num_classes=num_classes)
            elif cfg.model['net'].startswith('efficientnet'):
                net = EfficientNetBuilder(cfg.model['net'], num_classes=num_classes)
            elif cfg.model['net']=='RCNN':
                net = rcnn_32()
            elif cfg.model['net']=='pyramidnet272':
                net = PyramidNet_fastaugment(dataset=cfg.dataset,
                                             depth=272,
                                             alpha=200,
                                             num_classes=num_classes, 
                                             bottleneck=True)
            elif cfg.model['net']=='pyramidnet200': 
                net = PyramidNet('cifar100',200,240,100,bottleneck=True)
            elif model_config['net']=='densenet100':
                net = densenet_cifar()
            elif model_config['net']=='densenet_100_micronet':
                net= densenet_micronet(depth = 100, num_classes = 100, growthRate = 12, compressionRate = 2)
            elif model_config['net']=='densenet_172_micronet':
                net = densenet_micronet(depth = 172, num_classes = 100, growthRate = 30, compressionRate = 2)
                
            else:
                net = ResNet(cfg.model['net'], num_classes=num_classes)
        return net

    def get_num_params(self, display_all_modules=False):
        total_num_params = 0
        for n, p in self.net.named_parameters():
            num_params = 1
            for s in p.shape:
                num_params *= s
            if display_all_modules: print("{}: {}".format(n, num_params))
            total_num_params += num_params
        return total_num_params

    def _init_net(self):
        net = self._load_net()
        net = net.to(self.device)
        if self.device == 'cuda':
            net = nn.DataParallel(net)
            cudnn.benchmark = True
        self.summary['dataset'] = cfg.dataset
        self.summary['net'] = cfg.model['net']
        self.summary['num_params'] = self.get_num_params()
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
            self.summary['optimizer'] = 'SGD'
        elif self.mode == 'boosted':
            optimizer = RangerLars(self.net.parameters())
            self.summary['optimizer'] = 'Ranger + LARS'
        return optimizer

    def _init_scheduler(self):
        if self.mode == 'baseline':
            scheduler = ReduceLROnPlateau(self.optimizer,
                                          mode     = 'min',
                                          factor   = 0.2,
                                          patience = 20,
                                          verbose  = True)
            self.summary['scheduler'] = 'ROP'
        elif self.mode == 'alternative':
            half_train = cfg.train['nb_epochs']//2
            scheduler = DelayedCosineAnnealingLR(self.optimizer, half_train, half_train)
            self.summary['scheduler'] = 'Delayed Cosine Annealing'
        return scheduler

    def binary_connect(self, bits):
        """
        should be call only in train.py by the '_init_binary_connect' method.
        likewise, '_init_binary_connect' should never be called explicitely.
        The correct way to use this method is to modify
        the 'use_binary_connect' param of the train_config dict defined in config.py
        """
        return BC(self.net)
