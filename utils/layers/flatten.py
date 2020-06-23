import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)