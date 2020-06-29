""" Defines a simple Flatten layer """

import torch.nn as nn


class Flatten(nn.Module):
    """ Flatten a tensor in the following way:
    If input is x = torch.Tensor([d_1, d_2, ..., d_k]),
    returns     y = torch.Tensor([d_1, d_2*...*d_k])
    The returned tensor is always 2-dimensional.
    """
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
