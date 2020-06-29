""" Takes an Optimizer config dataclass (see config.py) and return an optimizer object,
used in model.py in the .configure_optimizers() method of the LightningModel class """

from torch.optim import SGD
from utils.model.optim import Ralamb, Lookahead


def init_optimizer(net, config):
    """ defines an optimizer in two steps:
        1. select an optimizer: SGD or RAlamb
        2. add LookAhead or not
    """
    if config.optim == 'SGD':
        optimizer = SGD(net.parameters(),
                        lr           = config.params['SGD']['lr'],
                        momentum     = config.params['SGD']['momentum'],
                        nesterov     = config.params['SGD']['nesterov'],
                        weight_decay = config.params['SGD']['weight_decay'])
    else:
        optimizer = Ralamb(net.parameters(),
                           lr           = config.params['RAlamb']['lr'],
                           betas        = config.params['RAlamb']['betas'],
                           eps          = config.params['RAlamb']['eps'],
                           weight_decay = config.params['RAlamb']['weight_decay'])
    if config.use_lookahead:
        return Lookahead(optimizer,
                         alpha = config.lookahead['alpha'],
                         k     = config.lookahead['k'])
    else:
        return optimizer
