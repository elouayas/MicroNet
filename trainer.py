import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils.decorators import *
from utils.models import *
from utils.tweak import EarlyStopping, Distillator
from utils.augment import Cutmix
from utils.score import score2019

from dataloader import get_dataloaders
from model import Model
from mask import Mask

import config as cfg

#import k_means
# TODO: add kmeans option in a clean way

class Trainer():
     
    @summary(cfg)
    def __init__(self, model):
        self.trainloader, self.testloader = get_dataloaders()
        """model must be an instance of the Model class"""
        self.model              = model
        self.binary_connect     = self._init_binary_connect()
        self.mask               = self._init_pruning()
        self.distillator        = self._init_distillation()
        #self.kmeans             = k_means.K_means(model.net)
        self.writer             = SummaryWriter(log_dir=cfg.log['tensorboard_path'])
        self.early_stopping     = EarlyStopping(cfg.train, cfg.log['checkpoints_path'])

    def _init_distillation(self):
        if not cfg.train['distillation']:
            return None
        teacher = Model(cfg.teacher['net'])
        teacher.load(cfg.teacher['teacher_path'])
        return Distillator(cfg.dataset, teacher, cfg.teacher)
         
    def _init_binary_connect(self):
        if cfg.train['use_binary_connect']:
            return self.model.binary_connect()

    def _init_pruning(self):
        if not cfg.train['use_pruning']:
            return None
        mask = Mask(self.model.net, cfg.pruning, cfg.train['nb_epochs'])
        mask.net = self.model.net
        mask.init_mask(0) # epoch number is 0 here
        mask.do_mask()
        self.model.net = self.mask.net
        return mask

    def infere(self, inputs, labels):
        r = np.random.rand(1)
        if cfg.train['use_cutmix'] and r < cfg.train['p']:
            lam, inputs, target_a, target_b = Cutmix(inputs, labels, cfg.train['beta'])
            outputs, layers = self.model.net(inputs)
            loss = self.model.criterion(outputs, target_a) * lam + \
                   self.model.criterion(outputs, target_b) * (1. - lam)
        else:
            outputs, layers = self.model.net(inputs) 
            loss = self.model.criterion(outputs, labels)
        return outputs, layers, loss

    def train(self):
        self.model.net.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(self.trainloader):
            # warmup cosine implementation based on batch iterations
            if cfg.scheduler['type'] == 'WarmupCosine': 
                self.model.scheduler(self.model.optimizer)
            inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
            if cfg.train['use_binary_connect']:
                self.binary_connect.binarization()
            inputs, labels = Variable(inputs), Variable(labels)
            #self.kmeans.save_params()
            self.model.optimizer.zero_grad()
            outputs, layers, loss = self.infere(inputs, labels)
            if cfg.train['distillation']:
                loss = self.distillator.run(inputs, outputs, labels, layers)
            loss.backward()
            if cfg.train['use_binary_connect']:
                self.binary_connect.clip()
            self.model.optimizer.step()
            #self.kmeans.restore(2,40)
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_acc = 100.*correct/total
        return train_loss, train_acc

    def test(self):
        self.model.net.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader): 
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                inputs, labels = Variable(inputs), Variable(labels)
                if cfg.train['use_binary_connect']:
                    self.binary_connect.binarization()
                outputs = self.model.net(inputs)[0]
                loss = self.model.criterion(outputs, labels)
                if cfg.train['use_binary_connect']:
                    self.binary_connect.clip()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100.*correct/total
        return test_loss, test_acc
    
    def prune(self, epoch):
        self.mask.net = self.model.net
        self.mask.verbose()
        self.mask.init_mask(epoch)
        self.mask.do_mask()
        self.mask.verbose()
        self.model.net = self.mask.net
    
    @verbose
    @toTensorboard
    def one_epoch_step(self, current_epoch, nb_epochs):
        """
            the two epoch params are used in the verbose decorator
            the returns values are used in verbose and toTensorboard
        """
        train_loss, train_acc = self.train()
        test_loss, test_acc = self.test()
        if cfg.scheduler['type'] == 'ROP':
            self.model.scheduler.step(test_loss)
        elif cfg.scheduler['type'] != 'WarmUpCosine':
            self.model.scheduler.step()
        if cfg.train['use_pruning']:
            self.prune(current_epoch)
        lr = self.model.optimizer.param_groups[0]['lr']
        return self.writer, train_loss, train_acc, test_loss, test_acc, lr

    @timed
    def run(self):
        for epoch in range(cfg.train['nb_epochs']):
            test_loss = self.one_epoch_step(epoch, cfg.train['nb_epochs'])[3]
            if cfg.train['use_early_stopping']:
                self.early_stopping(test_loss, self.model.net)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        self.model.net.eval()
        score2019(self.model)
        
        

    



