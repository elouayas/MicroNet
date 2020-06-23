from torch.optim.lr_scheduler import (ReduceLROnPlateau, MultiStepLR,
                                      CosineAnnealingLR, CosineAnnealingWarmRestarts)
from utils.schedulers import *


def init_scheduler(optimizer, config, batch_size, epochs):
    if config['type'] == 'ROP':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode     = config['params']['mode'],
                                      factor   = config['params']['factor'],
                                      patience = config['params']['patience'],
                                      verbose  = config['params']['verbose'])
    elif config['type'] == 'MultiStep':
        scheduler = MultiStepLR(optimizer,
                                milestones = config['params']['milestones'],
                                gamma      = config['params']['gamma'],
                                last_epoch = config['params']['last_epoch'])
    elif config['type'] == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, config['params']['epochs'])
    elif config['type'] == 'WarmRestartsCosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                T_0        = config['params']['T_0'],
                                T_mult     = config['params']['T_mult'],
                                eta_min    = config['params']['eta_min'],
                                last_epoch = config['params']['last_epoch'])
    else:
        scheduler = WarmupCosineLR(batch_size, 50000, epochs,
                                   config['params']['base_lr'], config['params']['target_lr'],
                                   config['params']['warm_up_epoch'], config['params']['cur_epoch'])
    if config['use_delay']:
        return DelayerScheduler(optimizer, config['delay']['delay_epochs'], scheduler)
    else:
        return scheduler