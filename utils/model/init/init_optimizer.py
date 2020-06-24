from torch.optim import SGD
from utils.model.optim import Ralamb, Lookahead


def init_optimizer(net, config):
    if config['type'] == 'SGD':
        optimizer = SGD(net.parameters(),
                        lr           = config['params']['lr'],
                        momentum     = config['params']['momentum'],
                        nesterov     = config['params']['nesterov'],
                        weight_decay = config['params']['weight_decay'])
    else:
        optimizer = Ralamb(net.parameters(),
                        lr           = config['params']['lr'],
                        betas        = config['params']['betas'],
                        eps          = config['params']['eps'],
                        weight_decay = config['params']['weight_decay'])
    if config['use_lookahead']:
        return Lookahead(optimizer,
                         alpha = config['lookahead']['alpha'],
                         k     = config['lookahead']['k'])
    else:
        return optimizer


