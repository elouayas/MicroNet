""" Takes a Model config dataclass (see config.py) and return a criterion object,
used in model.py to set the criterion attribute of the LightningModel class. """

from torch.nn import CrossEntropyLoss
from utils.model.layers import LabelSmoothingCrossEntropy

def init_criterion(config):
    """ returns the loss to be used by a LightningModel object,
        possibly using label smoothing.
    """
    if config.use_label_smoothing:
        return LabelSmoothingCrossEntropy(smoothing = config['smoothing'],
                                          reduction = config['reduction'])
    else:
        return CrossEntropyLoss()
