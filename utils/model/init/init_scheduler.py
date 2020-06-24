from torch.optim.lr_scheduler import (ReduceLROnPlateau, MultiStepLR,
                                      CosineAnnealingLR, CosineAnnealingWarmRestarts)
from utils.model.schedulers import GradualWarmupScheduler, DelayerScheduler


def init_scheduler(optimizer, config):
    """ Define LR scheduler in 3 steps:
            1. setup base LR scheduler
            2. optionally add warmup
            3. optionally add decay
        Oviously one must be careful not to mix things up 
        (decay + warmup, warmup + warmrestart ...) 
    """
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
        scheduler = CosineAnnealingLR(optimizer,
                                      config['params']['epochs'],
                                      eta_min    = config['params']['eta_min'],
                                      last_epoch = config['params']['last_epoch'])
    elif config['type'] == 'WarmRestartsCosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                T_0        = config['params']['T_0'],
                                T_mult     = config['params']['T_mult'],
                                eta_min    = config['params']['eta_min'],
                                last_epoch = config['params']['last_epoch'])
    if config['use_delay']:
        scheduler = DelayerScheduler(optimizer, config['delay']['delay_epochs'], scheduler)
    if config['use_warmup']:
        scheduler = GradualWarmupScheduler(optimizer,
                                           config['warmup']['multiplier'],
                                           config['warmup']['warmup_epochs'],
                                           scheduler)
    return scheduler