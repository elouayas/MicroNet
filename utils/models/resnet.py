'''
This program implements the ResNet architecture.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from utils.layers import Swish, Mish, SimpleSelfAttention, ShakeDrop

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


def noop(x):
    return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockSmall(nn.Module):
    ''' A basic block for small ResNet architectures. '''
    expansion = 1

    def __init__(self, in_planes, planes, activation, attention, sym, stride = 1, option = 'A',shkdrp = False , p_shakedrop = 1.0):
        super(BasicBlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 
                               kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        if attention:
            self.bn2 = nn.BatchNorm2d(planes)
            nn.init.constant_(self.bn2.weight, 0.)
            self.attention = SimpleSelfAttention(self.expansion * planes,ks=1,sym=sym)
        else:
            self.bn2 = nn.BatchNorm2d(planes) 
            self.attention = noop
        self.activation = self._init_activation(activation)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper, uses option A.
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], 
                                            (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes,
                               kernel_size = 1, stride = stride, bias = False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.shkdrp = shkdrp
        self.shake_drop = ShakeDrop(p_shakedrop)

        
    def _init_activation(self, activation):
        if activation == 'relu':
            return F.relu
        elif activation == 'Swish':
            return Swish()
        else:
            return Mish()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.attention(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = self.activation(out)
        if self.shkdrp:
            out = self.shake_drop(out)
        return out


class ResNetSmall(nn.Module):
    ''' Small ResNet architectures. '''
    
    def __init__(self, block, num_blocks, num_classes = 10,
                 activation='relu', attention=False, sym=False,shakedrop = False):
        super(ResNetSmall, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1,
                                       activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2,
                                       activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2,
                                       activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
        self.linear = nn.Linear(64, num_classes)
        self.activation = self._init_activation(activation)
        self.apply(_weights_init)

    def _init_activation(self, activation):
        if activation == 'relu':
            return F.relu
        elif activation == 'Swish':
            return Swish()
        else:
            return Mish()

    def _make_layer(self, block, planes, num_blocks, stride, activation, attention, sym, shakedrop):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, activation, attention, sym, stride, shkdrp=shakedrop))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        pool1 = out
        out = self.layer2(out)
        pool2 = out
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        pool3 = out
        out = self.linear(out)
        return out, [pool1, pool2, pool3]


def ResNet(arch, num_classes = 100, activation='relu', attention=False, sym=False, shakedrop=False):
    ''' Constructs a ResNet model. '''
    if arch == 'resnet20':
        return ResNetSmall(BasicBlockSmall, [3, 3, 3], num_classes,
                           activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
    elif arch == 'resnet32':
        return ResNetSmall(BasicBlockSmall, [5, 5, 5], num_classes,
                           activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
    elif arch == 'resnet44':
        return ResNetSmall(BasicBlockSmall, [7, 7, 7], num_classes,
                           activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
    elif arch == 'resnet56':
        return ResNetSmall(BasicBlockSmall, [9, 9, 9], num_classes,
                           activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
    elif arch == 'resnet110':
        return ResNetSmall(BasicBlockSmall, [18, 18, 18], num_classes,
                           activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)
    elif arch == 'resnet1202':
        return ResNetSmall(BasicBlockSmall, [200, 200, 200], num_classes,
                           activation=activation, attention=attention, sym=sym, shakedrop=shakedrop)






