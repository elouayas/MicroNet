""" Swish activation function from https://arxiv.org/abs/1710.05941 """

import torch
import torch.nn as nn

class Swish(nn.Module):
    """ inplace Swish activation function """

    def forward(self, x):
        return x * torch.sigmoid(x)
