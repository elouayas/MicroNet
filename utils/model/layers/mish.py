""" See https://arxiv.org/abs/1908.08681 """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):

    """ inline Mish activation function """

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))
