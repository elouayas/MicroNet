from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils.models import *
from utils.tweak import EarlyStopping, Distillator
from utils.decorators import timed, summary
from utils.score import score2019

from dataloader import get_dataloaders
from model import Model
from mask import Mask

import config as cfg


class Trainer():
     
    @summary(cfg.dataset, cfg.model, cfg.train)
    def __init__(self, model):
        self.model  = model # model must be an instance of the Model class
        self.trainloader, self.testloader = get_dataloaders()
        if cfg.train['distillation']:
            self.teacher_config = cfg.teacher
        self.writer         = SummaryWriter(log_dir=cfg.log['tensorboard_path'])
        self.early_stopping = EarlyStopping(cfg.train, cfg.log['checkpoints_path'])
        self.use_binary_connect = self._init_binary_connect()
        self.use_pruning = self._init_pruning()
        self.state = {'train_loss': 0, 'train_acc': 0, 'test_loss': 0, 'test_acc': 0, 
                      'best_acc': 0, 'epoch': 0, 'score_param': 1,'score_op': 1,
                      'lr': self.model.optimizer.param_groups[0]['lr']}
         
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

    def train(self, distillator):
        self.model.net.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, labels in tqdm(self.trainloader):
            inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
            if self.use_binary_connect:
                self.binary_connect.binarization()
            inputs, labels = Variable(inputs), Variable(labels)
            self.model.optimizer.zero_grad()
            if cfg.train['distillation']:
                #outputs, layers = self.model.net(inputs)#### a changer
                outputs = self.model.net(inputs)#### a changer
                loss = distillator.run(inputs, outputs, labels)
            else:
                outputs = self.model.net(inputs)
                loss = self.model.criterion(outputs, labels)
            loss.backward()
            if self.use_binary_connect:
                self.binary_connect.clip()
            self.model.optimizer.step()
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
    
    
    def verbose(self):
        print()
        print('Train Loss................: {:.2f}'.format(self.state['train_loss']))
        print('Test Loss.................: {:.2f}'.format(self.state['test_loss']))
        print('Train Accuracy............: {:.2f}'.format(self.state['train_acc']))
        print('Test Accuracy.............: {:.2f}'.format(self.state['test_acc']))
        print()
        print('Current Learning Rate.....: {:.10f}'.format(self.state['lr']))
        print('Best Test Accuracy........: {:.2f}'.format(self.state['best_acc']))
        
    
    def to_tensorbard(self, epoch):
        self.writer.add_scalar('Loss/train', self.state['train_loss'], epoch)
        self.writer.add_scalar('Accuracy/train', self.state['train_acc'], epoch)
        self.writer.add_scalar('Loss/test', self.state['test_loss'], epoch)
        self.writer.add_scalar('Accuracy/test', self.state['test_acc'], epoch)
        self.writer.add_scalar('Learning Rate/lr', self.state['lr'], epoch)
        
        
    def update_state(self, epoch, distillator):
        train_loss, train_acc = self.train(distillator)
        test_loss, test_acc = self.test()
        if cfg.model['mode'] == 'basic':
            self.model.scheduler.step(test_loss) # ROP
        else:
            self.model.scheduler.step()
        self.state['train_loss'] = train_loss
        self.state['train_acc']  = train_acc
        self.state['test_loss']  = test_loss
        self.state['test_acc']   = test_acc
        self.state['lr']         = self.model.optimizer.param_groups[0]['lr']
        if test_acc > self.state['best_acc']:
            self.state['best_acc'] = test_acc

            
    def one_epoch_step(self, epoch, distillator):
        print(80*'_')
        print('EPOCH %d / %d' % (epoch+1, cfg.train['nb_epochs']))
        self.update_state(epoch, distillator)
        self.to_tensorbard(epoch)
        if cfg.train['verbose']:
            self.verbose()
        if self.use_pruning:
            self.prune(epoch)
    

    @timed
    def run(self):
        if  cfg.train['distillation']:
            teacher = Model(cfg.teacher['net'])
            teacher.load(cfg.teacher['teacher_path'])
            distillator = Distillator(cfg.dataset, teacher, cfg.teacher)
        else : 
            distillator = None 
        for epoch in range(cfg.train['nb_epochs']):
            self.one_epoch_step(epoch, distillator)
            if cfg.train['use_early_stopping']:
                self.early_stopping(self.state['test_loss'], self.model.net)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        score2019(self.model)
        
        

    



