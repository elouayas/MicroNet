"""
Model Class for Training Pipeline.
Used as the model attribute of the Trainer Class.
"""

import math

import torch

from utils.init import *
from utils.decorators import timed, saveWrapper
from utils.layers import Mish
from utils.tweak import BinaryConnect

import config as cfg


class Model():
    
    @timed
    def __init__(self, net):
        self.device     = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.net        = init_net(cfg.dataset, net, self.device, cfg.model)
        self.optimizer  = init_optimizer(self.net, cfg.optim)
        self.scheduler  = init_scheduler(self.optimizer, cfg.scheduler,
                                         cfg.dataloader['train_batch_size'], cfg.train['nb_epochs'])
        self.criterion  = init_criterion(cfg.model)
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

    def binary_connect(self, bits):
        return BinaryConnect(self.net)

    def load(self, path, weight_only=True):
        if weight_only:
            self.net.load_state_dict(torch.load(path))
        else:
            checkpoints = torch.load(path)
            self.net.load_state_dict(checkpoints['state_dict'])

    #TODO: add train infos to the dicts past to torch.save
    @saveWrapper
    def save(self, interrupted = False):
        checkpoints_path = './checkpoints/' + cfg.dataset + '/'
        name = cfg.get_experiment_name()
        if interrupted:
            filename = 'INTERRUPTED_' + name + '.pt'
        else:
            filename = name + '.pt'
        torch.save(self.net.state_dict(), checkpoints_path + filename)
        torch.save({
                'state_dict':self.state_dict()
            }, checkpoints_path + filename)