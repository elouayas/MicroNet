import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils.models import *
from utils.tweak import EarlyStopping, Distillator
from utils.augment import Cutmix
from utils.decorators import *
from utils.score import score2019

from dataloader import get_dataloaders
from model import Model
from mask import Mask

import config as cfg

import k_means


class Trainer():
     
    @summary(cfg.dataset, cfg.model, cfg.train)
    def __init__(self, model):
        self.trainloader, self.testloader = get_dataloaders()
        self.model              = model # model must be an instance of the Model class
        self.writer             = SummaryWriter(log_dir=cfg.log['tensorboard_path'])
        self.early_stopping     = EarlyStopping(cfg.train, cfg.log['checkpoints_path'])
        self.use_binary_connect = self._init_binary_connect()
        self.use_pruning        = self._init_pruning()
        self.kmeans             = k_means.K_means(model.net)
         
    def _init_binary_connect(self):
        if cfg.train['use_binary_connect']:
            self.binary_connect = self.model.binary_connect()
            return True
        else:
            return False
    
    def _init_pruning(self):
        if cfg.train['use_pruning']:
            self.mask = Mask(self.model.net, cfg.pruning, cfg.train['nb_epochs'])
            self.mask.net = self.model.net
            self.mask.init_mask(0) # epoch number is 0 here
            self.mask.do_mask()
            self.model.net = self.mask.net
            return True
        else:
            return False

    # TODO: add kmeans option in a clean way
    def train(self, distillator):
        self.model.net.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(self.trainloader):
            # self.model.scheduler(self.model.optimizer)
            inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
            if self.use_binary_connect:
                self.binary_connect.binarization()
            inputs, labels = Variable(inputs), Variable(labels)
            #self.kmeans.save_params()
            self.model.optimizer.zero_grad()
            #outputs, layers = self.model.net(inputs)#### a changer
            r = np.random.rand(1)
            if cfg.train['use_cutmix'] and r < cfg.train['p']:
                lam, inputs, target_a, target_b = Cutmix(inputs, labels, cfg.train['beta'])
                outputs = self.model.net(inputs) #### a changer
                loss = self.model.criterion(outputs, target_a) * lam + \
                       self.model.criterion(outputs, target_b) * (1. - lam)
            else:
                outputs = self.model.net(inputs) #### a changer
                loss    = self.model.criterion(outputs, labels) 
            if cfg.train['distillation']:
                loss = distillator.run(inputs, outputs, labels)
            loss.backward()
            if self.use_binary_connect:
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
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.testloader): 
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                inputs, labels = Variable(inputs), Variable(labels)
                if self.use_binary_connect:
                    self.binary_connect.binarization()
                outputs = self.model.net(inputs)
                loss = self.model.criterion(outputs, labels)
                if self.use_binary_connect:
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
    def one_epoch_step(self, current_epoch, nb_epochs, distillator):
        """
            the two epoch params are used in the verbose decorator
            the returns values are used in verbose and to_tensorboard
        """
        train_loss, train_acc = self.train(distillator)
        test_loss, test_acc = self.test()
        if cfg.model['mode'] == 'basic':
            self.model.scheduler.step(test_loss) # ROP
        else:
            self.model.scheduler.step()
        if self.use_pruning:
            self.prune(current_epoch)
        lr = self.model.optimizer.param_groups[0]['lr']
        return self.writer, train_loss, train_acc, test_loss, test_acc, lr
    

    @timed
    def run(self):
        if  cfg.train['distillation']:
            teacher = Model(cfg.teacher['net'])
            teacher.load(cfg.teacher['teacher_path'])
            distillator = Distillator(cfg.dataset, teacher, cfg.teacher)
        else : 
            distillator = None 
        for epoch in range(cfg.train['nb_epochs']):
            _, _, _, test_loss, _, _ = self.one_epoch_step(epoch, 
                                                           cfg.train['nb_epochs'],
                                                           distillator)
            if cfg.train['use_early_stopping']:
                self.early_stopping(test_loss, self.model.net)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        # self.model.net.eval()
        # score2019(self.model)
        
        

    



