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
# from efficientnet_pytorch import EfficientNet


from utils.binary_connect import BC
from utils.label_smoothing import LabelSmoothingCrossEntropy
from utils.optim.delayed_lr_scheduler import DelayedCosineAnnealingLR
from utils.optim.ranger_lars import RangerLars
from utils.decorators import timed
from utils.mish import Mish

class Model():
    @timed
    def __init__(self, model_config, dataloader_config, dataset):
        self.device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.mode       = model_config['mode']
        self.summary    = {} # will contain dataset, net, num_params, optimizer, scheduler
        self.net        = self._init_net(model_config, dataset)
        self.update_activation_function(model_config)
        self.criterion  = self._init_criterion(model_config)
        self.optimizer  = self._init_optimizer()
        self.scheduler  = self._init_scheduler()

    def _load_net(self, model_config, dataset):
        num_classes = 100 if dataset=='cifar100' else 10
        if model_config['quantize']:
            if model_config['net']=='wide_resnet_28_10':
                net = WideResNet_28_10_Quantized(num_classes=num_classes)
            else:
                net = ResNet_Quantized(model_config['net'], num_classes=num_classes)
        else:
            if model_config['net']=='wide_resnet_28_10':
                net = WideResNet_28_10(num_classes=num_classes)
            elif model_config['net'].startswith('efficientnet'):
                net = EfficientNetBuilder(model_config['net'], num_classes=num_classes)
            elif model_config['net']=='RCNN':
                net = rcnn_32()
            else:
                net = ResNet(model_config['net'], num_classes=num_classes)
        return net

    def get_num_params(self, net, display_all_modules=False):
        total_num_params = 0
        for n, p in net.named_parameters():
            num_params = 1
            for s in p.shape:
                num_params *= s
            if display_all_modules: print("{}: {}".format(n, num_params))
            total_num_params += num_params
        return total_num_params

    def _init_net(self, model_config, dataset):
        net = self._load_net(model_config, dataset)
        net = net.to(self.device)
        if self.device == 'cuda':
            net = nn.DataParallel(net)
            cudnn.benchmark = True
        self.summary['dataset'] = dataset
        self.summary['net'] = model_config['net']
        self.summary['num_params'] = self.get_num_params(net)
        return net

    def update_activation_function(self, model_config):
        if model_config['activation'] == 'ReLU':
            pass
        else:
            for name, module in self.net._modules.items():
                if isinstance(module, nn.ReLU):
                    self.net._modules[name] = Mish

    def _init_criterion(self, model_config):
        if model_config['label_smoothing']:
            return LabelSmoothingCrossEntropy(smoothing=model_config['smoothing'],
                                              reduction=model_config['reduction'])
        else:
            return CrossEntropyLoss()

    def _init_optimizer(self):
        if self.mode == 'baseline':
            optimizer = SGD(self.net.parameters(),
                            lr           = 0.1,
                            momentum     = 0.9,
                            nesterov     = False,
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
        elif self.mode == 'boosted':
            scheduler = DelayedCosineAnnealingLR(self.optimizer, 100, 100)
            self.summary['scheduler'] = 'Delayed Cosine Annealing'
        return scheduler

    # should be call only in train.py by the '_init_binary_connect' method.
    # likewise, '_init_binary_connect' should never be called explicitely.
    # The correct way to use this method is to modify
    # the 'use_binary_connect' param of the train_config dict defined in config.py
    def binary_connect(self, bits):
        return BC(self.net)
