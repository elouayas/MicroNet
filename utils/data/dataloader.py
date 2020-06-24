from utils.data.dataset import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

def get_train_dataloader(dataset, config):
    if dataset == 'cifar100':
        trainset = CIFAR100(config['rootdir'],
                            train    = True, 
                            scale    = 255,
                            policies = config['policies'],
                            augnum   = config['augnum'])
    else:
        trainset = CIFAR10(config.dataloader['rootdir'],
                           train    = True, 
                           scale    = 255,
                           policies = config['policies'],
                           augnum   = config['augnum'])
    return DataLoader(trainset,
                        batch_size  = config['train_batch_size'],
                        shuffle     = True,
                        num_workers = config['nb_workers'])

def get_val_dataloader(dataset, config):
    if dataset == 'cifar100':
        valset = CIFAR100(config['rootdir'], train = False, scale = 255)
    else:
        valset = CIFAR10( config['rootdir'], train = False, scale = 255)
    return DataLoader(valset,
                        batch_size  = config['val_batch_size'],
                        shuffle     = False,
                        num_workers = config['nb_workers'])