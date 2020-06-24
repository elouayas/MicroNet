"""
Adapted from https://github.com/wps712/MicroNetChallenge/blob/cifar100/models/cifar/densenet.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                     BOTTLENECK                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Bottleneck(nn.Module):
    
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1        = nn.BatchNorm2d(inplanes)
        self.conv1      = nn.Conv2d(inplanes, planes,     kernel_size=1, bias=False)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(  planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.dropRate   = dropRate
        
    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                      BASIC BLOCK                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class BasicBlock(nn.Module):
    
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1        = nn.BatchNorm2d(inplanes)
        self.conv1      = nn.Conv2d(inplanes, growthRate, kernel_size=3, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        self.dropRate   = dropRate
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                       TRANSITION                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Transition(nn.Module):
    
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1        = nn.BatchNorm2d(inplanes)
        self.conv1      = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                       DENSENET                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class DenseNet(nn.Module):

    def __init__(self, num_classes=100, block=Bottleneck, 
                 depth=22, dropRate=0, growthRate=12, compressionRate=2):
        super(DenseNet, self).__init__()
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6
        self.growthRate = growthRate
        self.dropRate   = dropRate

        # self.inplanes is a global variable used across multiple helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn         = nn.BatchNorm2d(self.inplanes)
        self.activation = nn.ReLU(inplace=True)
        self.avgpool    = nn.AvgPool2d(8)
        self.fc         = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks):
        layers = []
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate
        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes)

    def forward(self, x):
        "the pool variables are used by the graph distillation (GKD)"
        x = self.conv1(x)
        x = self.trans1(self.dense1(x)) 
        pool1 = x
        x = self.trans2(self.dense2(x))
        pool2 = x 
        x = self.dense3(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pool3 = x
        x = self.fc(x)
        return x, [pool1, pool2, pool3]

    @classmethod
    def from_name(cls, dataset, name):
        "create a densenet model according to the name"
        num_classes = 10 if dataset == 'cifar10' else 100 
        if   name == "densenet100":
            return cls(num_classes = num_classes,
                       depth = 100, growthRate = 12, compressionRate = 2)
        elif name == "densenet172":
            return cls(num_classes = num_classes,
                       depth = 172, growthRate = 30, compressionRate = 2)
