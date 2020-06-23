from torch.nn import CrossEntropyLoss
from utils.layers import LabelSmoothingCrossEntropy

def init_criterion(config):
    if config['label_smoothing']:
        return LabelSmoothingCrossEntropy(smoothing=config['smoothing'],
                                            reduction=config['reduction'])
    else:
        return CrossEntropyLoss()