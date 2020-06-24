from torch.nn import CrossEntropyLoss
from utils.model.layers import LabelSmoothingCrossEntropy

def init_criterion(config):
    if config['use_label_smoothing']:
        return LabelSmoothingCrossEntropy(smoothing = config['smoothing'],
                                          reduction = config['reduction'])
    else:
        return CrossEntropyLoss()