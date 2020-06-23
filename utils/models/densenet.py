import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.layers import Swish, Mish, SimpleSelfAttention, ShakeDrop

__all__ = ['densenet_micronet']

def noop(x):
    return x


class Bottleneck(nn.Module):
    def __init__(self, inplanes, expansion=4, growthRate=12, dropRate=0,
                 activation='relu', attention=False, sym=False, shkdrp = False , p_shakedrop = 1.0):
        super(Bottleneck, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if attention:
            self.bn2 = nn.BatchNorm2d(planes)
            nn.init.constant_(self.bn2.weight, 0.)
            self.attention = SimpleSelfAttention(self.expansion * planes,ks=1,sym=sym)
        else:
            self.bn2 = nn.BatchNorm2d(planes) 
            self.attention = noop
        self.conv2 = nn.Conv2d(planes, growthRate, kernel_size=3, padding=1, bias=False)
        self.activation = self._init_activation(activation)
        self.shkdrp = shkdrp
        self.shake_drop = ShakeDrop(p_shakedrop)
        self.dropRate = dropRate
        

    def _init_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'Swish':
            return Swish()
        else:
            return Mish()

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.attention(self.bn2(out))
        out = self.activation(out)
        out = self.conv2(out)
        if self.shkdrp:
            out = self.shake_drop(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, expansion=1, growthRate=12, dropRate=0, activation='relu',shkdrp = False , p_shakedrop = 1.0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3, padding=1, bias=False)
        self.activation = self._init_activation(activation)
        self.shkdrp = shkdrp
        self.shake_drop = ShakeDrop(p_shakedrop)
        self.dropRate = dropRate
    
    def _init_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'Swish':
            return Swish()
        else:
            return Mish()

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        if self.shkdrp:
            out = self.shake_drop(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, activation):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.activation = self._init_activation(activation)

    def _init_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'Swish':
            return Swish()
        else:
            return Mish()

    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet_MicroNet(nn.Module):

    def __init__(self, depth=22, block=Bottleneck, 
                 dropRate=0, num_classes=100, growthRate=12, compressionRate=2,
                 init = 'Default', activation='relu', attention=False, sym=False, shakedrop = False):
        super(DenseNet_MicroNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) / 3 if block == BasicBlock else (depth - 4) // 6

        self.growthRate = growthRate
        self.dropRate = dropRate
        self.shakedrop = shakedrop

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_denseblock(block, n, activation, attention, sym)
        self.trans1 = self._make_transition(compressionRate, activation)
        self.dense2 = self._make_denseblock(block, n, activation, attention, sym)
        self.trans2 = self._make_transition(compressionRate, activation)
        self.dense3 = self._make_denseblock(block, n, activation, attention, sym)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.activation = self._init_activation(activation)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _init_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'Swish':
            return Swish()
        else:
            return Mish()

    def _make_denseblock(self, block, blocks, activation, attention, sym):
        layers = []
        for i in range(blocks):
            
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, growthRate=self.growthRate, dropRate=self.dropRate,
                                activation=activation, attention=attention, sym=sym, shkdrp=self.shakedrop))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, activation):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, activation)


    def forward(self, x):
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



def densenet_micronet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return DenseNet_MicroNet(**kwargs)