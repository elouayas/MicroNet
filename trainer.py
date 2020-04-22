from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from utils.early_stopping import EarlyStopping
from utils.decorators import timed
from utils.score.score import score2019
from utils.distillation import *

from dataloader import get_dataloaders
from mask import Mask
from models import *

import config as cfg

class Trainer():
     
    def __init__(self, model):
        self.model = model # model must be an instance of the Model class
        self.trainloader, self.testloader = get_dataloaders()
        self.config = cfg.train_config
        if self.config['distillation']:
            self.teacher_config = cfg.teacher
        self.writer = torch.utils.tensorboard.SummaryWriter(log_dir=cfg.log['tensorboard_path'])
        self.early_stopping = self._init_early_stopping()
        self.use_binary_connect = self._init_binary_connect()
        self.use_pruning = self._init_pruning()
        self.state = {'train_loss': 0, 'train_acc': 0, 'test_loss': 0, 'test_acc': 0, 
                      'best_acc': 0, 'epoch': 0, 'score_param': 1,'score_op': 1}
        
        
    def __str__(self): 
        title = 'Training settings  :' + '\n' + '\n'
        dataset     = 'Dataset...................:  ' + self.model.summary['dataset'] + '\n'
        net         = 'Net.......................:  ' + self.model.summary['net'] + '\n' 
        num_params  = 'Number of parameters......:  {:.2e}'.format(self.model.summary['num_params']) + '\n'
        optimizer   = 'Optimizer.................:  ' + self.model.summary['optimizer'] + '\n'
        scheduler   = 'Learning Rate Scheduler...:  ' + self.model.summary['scheduler'] + '\n'
        nb_epochs   = 'Number of epochs..........:  ' + str(self.config['nb_epochs']) + '\n'
        use_bc      = 'Use Binary Connect........:  ' + str(self.use_binary_connect) + '\n'
        use_pruning = 'Use Soft Pruning..........:  ' + str(self.use_pruning) + '\n'
        model_summary = dataset + net + num_params + optimizer + scheduler
        train_summary = nb_epochs + use_bc + use_pruning
        return (80*'_' + '\n' + title + model_summary + train_summary + 80*'_')
         

    def _init_early_stopping(self):
        early_stopping = EarlyStopping(patience=self.config['patience'], 
                                       delta = self.config['delta'], 
                                       verbose=self.config['verbose'])
        early_stopping.set_checkpoints(self.model.summary, 
                                       self.config['checkpoints_path'])
        return early_stopping
    
    
    def _init_binary_connect(self):
        if self.config['use_binary_connect']:
            self.binary_connect = self.model.binary_connect()
            return True
        else:
            return False
    
    
    def _init_pruning(self):
        pruning_config = self.config['pruning']
        if pruning_config['use_pruning']:
            self.mask = Mask(self.model.net, pruning_config, self.config['nb_epochs'])
            self.mask.net = self.model.net
            self.mask.init_mask(0) # epoch number is 0 here
            self.mask.do_mask()
            self.model.net = self.mask.net
            return True
        else:
            return False
    

                   
    def train(self,teacher):
        if not self.config['distillation']:
            self.model.net.train()
            train_loss, correct, total = 0, 0, 0
            for inputs, labels in tqdm(self.trainloader):
                inputs, labels = inputs.to(self.model.device), labels.to(self.model.device)
                if self.use_binary_connect:
                    self.binary_connect.binarization()
                self.model.optimizer.zero_grad()
                inputs, labels = Variable(inputs), Variable(labels)
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
        else :
            self.model.net.train()
            train_loss, correct, total = 0, 0, 0
            criterion = BatchMeanCrossEntropyWithLogSoftmax()
            for inputs, targets in tqdm(self.trainloader):
                inputs, targets = Variable(inputs), Variable(targets)
                inputs, targets = inputs.to(self.model.device), targets.to(self.model.device)
                targets2 = to_one_hot(targets, 100) ### A CHANGER LE 100
                intra_class = torch.matmul(targets2,targets2.T)
                inter_class = 1 - intra_class
                targets = targets2.argmax(dim=1)
                self.model.optimizer.zero_grad()
                #outputs, layers = self.model.net(inputs)#### a changer
                outputs = self.model.net(inputs)#### a changer
                loss = criterion(F.log_softmax(outputs,dim=-1),targets2)

                with torch.no_grad():
                    #teacher_output, teacher_layers = teacher(inputs) #### a changer
                    teacher_output = teacher(inputs) #### a changer

                if self.teacher_config['lambda_hkd'] > 0:
                    p = F.softmax(teacher_output/self.teacher_config['temp'],dim=-1)
                    log_q = F.log_softmax(outputs/self.teacher_config['temp'],dim=-1)
                    log_p = F.log_softmax(teacher_output/self.teacher_config['temp'],dim=-1)
                    hkd_loss = BatchMeanKLDivWithLogSoftmax()(p=p,log_q=log_q,log_p=log_p)
                    loss += self.teacher_config['lambda_hkd']*hkd_loss
                if self.teacher_config['lambda_rkd']> 0:
                    loss_rkd = 0
                    zips = zip(layers,teacher_layers) if not self.teacher_config['pool3_only'] else zip([layers[-1]],[teacher_layers[-1]])
                    for student_layer,teacher_layer in zips:

                        distances_teacher = get_distances(teacher_layer)
                        distances_teacher = distances_teacher[distances_teacher>0]
                        mean_teacher = distances_teacher.mean()
                        distances_teacher = distances_teacher/mean_teacher
                            
                        distances_student = get_distances(student_layer)
                        distances_student = distances_student[distances_student>0]
                        mean_student = distances_student.mean()
                        distances_student = distances_student/mean_student
                        loss_rkd += self.teacher_config['lambda_rkd']*F.smooth_l1_loss(distances_student, distances_teacher, reduction='none').mean()
                    loss += loss_rkd if self.teacher_config['pool3_only'] else loss_rkd/3
                elif self.teacher_config['lambda_gkd'] > 0:
                    loss_gkd = do_gkd(self.teacher_config['pool3_only'], layers, teacher_layers, self.teacher_config['k'], self.teacher_config['power'], self.teacher_config['intra_only'], self.teacher_config['lambda_gkd'], intra_class, self.teacher_config['inter_only'], inter_class)
                    loss += loss_gkd if self.teacher_config['pool3_only'] else loss_gkd/3
                loss.backward()
                self.model.optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
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
        print('Current Learning Rate.....: {:.10f}'.format(self.model.optimizer.param_groups[0]['lr']))
        print('Best Test Accuracy........: {:.2f}'.format(self.state['best_acc']))
        
    
    def to_tensorbard(self, epoch):
        self.writer.add_scalar('Loss/train', self.state['train_loss'], epoch)
        self.writer.add_scalar('Accuracy/train', self.state['train_acc'], epoch)
        self.writer.add_scalar('Loss/test', self.state['test_loss'], epoch)
        self.writer.add_scalar('Accuracy/test', self.state['test_acc'], epoch)
        self.writer.add_scalar('Learning Rate/lr', self.model.optimizer.param_groups[0]['lr'], epoch)
        
        
    def update_state(self, epoch,teacher):
        train_loss, train_acc = self.train(teacher)
        test_loss, test_acc = self.test()
        if self.model.summary['scheduler']=='ROP':
            self.model.scheduler.step(test_loss)
        else:
            self.model.scheduler.step()
        self.state['train_loss'] = train_loss
        self.state['train_acc']  = train_acc
        self.state['test_loss'] = test_loss
        self.state['test_acc'] = test_acc
        if test_acc > self.state['best_acc']:
            self.state['best_acc'] = test_acc

            
    def one_epoch_step(self, epoch,teacher):
        print(80*'_')
        print('EPOCH %d / %d' % (epoch+1, self.config['nb_epochs']))
        self.update_state(epoch,teacher)
        self.to_tensorbard(epoch)
        if self.config['verbose']:
            self.verbose()
        if self.use_pruning:
            self.prune(epoch)
    

    @timed
    def run(self):
        if  self.config['distillation']:
            ######################### A CHANGER EN UTILISANT LA CLASSE MODEL PROPREMENT
            path = self.teacher_config['teacher_path']
            checkpoint = torch.load(path)
            teacher =torch.nn.DataParallel(PyramidNet('cifar100',200,240,100,bottleneck=True)) # à changer
            teacher.load_state_dict(checkpoint['state_dict'])
            #teacher = torch.nn.DataParallel(PyramidNet_fastaugment(dataset='cifar100',depth=272,alpha=200,num_classes=100, bottleneck=True))
            #teacher.load_state_dict(checkpoint['model'])
            
        else : 
            teacher = 'None'
        for epoch in range(self.config['nb_epochs']):
            self.one_epoch_step(epoch,teacher)
            if self.config['use_early_stopping']:
                self.early_stopping(self.state['test_loss'], self.model.net)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break
        score2019(self.model)
        
        

    



