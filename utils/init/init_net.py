import torch.nn as nn
from torch.backends import cudnn

from utils.models import *

#TODO: add attentio params to other nets.

def init_net(dataset, net_name, device, config):    
    num_classes = 100 if dataset == 'cifar100' else 10
    if net_name == 'wide_resnet_28_10':
        net = WideResNet_28_10(num_classes = num_classes, 
                                       activation  = config['activation'])
    elif net_name.startswith('efficientnet'):
        net = EfficientNetBuilder(net_name, num_classes = num_classes)
    elif net_name == 'densenet_final':
        net = densenet_final(depth           = 22, 
                                num_classes     = 100, 
                                growthRate      = 12, 
                                compressionRate = 2)
    elif net_name == 'densenet22':
        net = densenet_micronet(depth           = 22, 
                                num_classes     = 100, 
                                growthRate      = 12, 
                                compressionRate = 2)
    elif net_name =='densenet100':
        net = densenet_micronet(depth           = 100, 
                                num_classes     = 100, 
                                growthRate      = 12, 
                                compressionRate = 2,
                                activation      = config['activation'],
                                attention       = config['self_attention'],
                                sym             = config['attention_sym'],
                                shakedrop       = config['shakedrop'])
    elif net_name == 'densenet172':
        net = densenet_micronet(depth           = 172, 
                                num_classes     = 100, 
                                growthRate      = 30, 
                                compressionRate = 2,
                                activation      = config['activation'],
                                attention       = config['self_attention'],
                                sym             = config['attention_sym'],
                                shakedrop       = config['shakedrop'])
    else:
        net = ResNet(net_name, 
                    num_classes = num_classes,
                    activation  = config['activation'],
                    attention   = config['self_attention'],
                    sym         = config['attention_sym'],
                    shakedrop   = config['shakedrop'])
    net = net.to(device)
    if device == 'cuda:0':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    return net