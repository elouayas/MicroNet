""" Takes a Scheduler config dataclass (see config.py) and return a scheduler object,
    used in model.py in the .configure_optimizers() method of the LightningModel class """

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
    if config.scheduler == 'ROP':
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode     = config.params['ROP']['mode'],
                                      factor   = config.params['ROP']['factor'],
                                      patience = config.params['ROP']['patience'],
                                      verbose  = config.params['ROP']['verbose'])
    elif config.scheduler == 'MultiStep':
        scheduler = MultiStepLR(optimizer,
                                milestones = config.params['MultiStep']['milestones'],
                                gamma      = config.params['MultiStep']['gamma'],
                                last_epoch = config.params['MultiStep']['last_epoch'])
    elif config.scheduler == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max      = config.params['Cosine']['epochs'],
                                      eta_min    = config.params['Cosine']['eta_min'],
                                      last_epoch = config.params['Cosine']['last_epoch'])
    elif config.scheduler == 'WarmRestartsCosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                T_0        = config.params['WarmRestartsCosine']['T_0'],
                                                T_mult     = config.params['WarmRestartsCosine']['T_mult'],
                                                eta_min    = config.params['WarmRestartsCosine']['eta_min'],
                                                last_epoch = config.params['WarmRestartsCosine']['last_epoch'])
    if config.use_delay:
        scheduler = DelayerScheduler(optimizer, config.delay['delay_epochs'], scheduler)
    if config.use_warmup:
        scheduler = GradualWarmupScheduler(optimizer,
                                           config.warmup['multiplier'],
                                           config.warmup['warmup_epochs'],
                                           scheduler)
    return scheduler
