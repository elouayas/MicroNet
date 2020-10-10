""" Defines specifig loss to be used in distillation.
    See utils.distillation
"""

import torch.nn as nn


class BatchMeanCrossEntropyWithLogSoftmax(nn.Module):
    """ used in utils.distillation  """

    def forward(self, y_hat, y):
        return -(y_hat*y).sum(dim=1).mean(dim=0)

class BatchMeanKLDivWithLogSoftmax(nn.Module):
    """ used in utils.distillation  """

    def forward(self, p, log_q,  log_p):
        return (p*log_p - p*log_q).sum(dim=1).mean(dim=0)
