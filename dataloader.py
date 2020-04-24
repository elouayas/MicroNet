from PIL import Image

from torchvision import transforms
from torchvision.datasets import CIFAR10,CIFAR100
from torch.utils.data import DataLoader

from utils.augment import CIFAR10Policy
from utils.augment.cutout import Cutout
from utils.augment.fastaugmentations import Augmentation
from utils.augment.archive import fa_reduced_cifar10

import config as cfg


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                       TRANSFORMS                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def get_transforms():
    if cfg.dataset=='cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else: #cifar100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    normalize = [transforms.Normalize(mean, std)]
    to_tensor = [transforms.ToTensor()]
    if cfg.dataloader['resize']:
        resize = [transforms.Resize((224,224), interpolation=Image.BICUBIC)]
    else:
        resize = []
    if cfg.dataloader['use_cutout']:
        cutout = [Cutout(cfg.dataloader['n_holes'],cfg.dataloader['length'])]
    else:
        cutout = []
    aug = [transforms.RandomCrop(32, padding=4), 
           transforms.RandomHorizontalFlip()]
    if cfg.dataloader['data_aug']:
        aug += [CIFAR10Policy()]
    transform_train = transforms.Compose(aug + resize + to_tensor + cutout + normalize)
    transform_test  = transforms.Compose(resize + to_tensor + normalize)
    if cfg.dataloader['fast_aug']:
        transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
    return transform_train, transform_test



# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                       DATASETS                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def get_datasets(transform_train, transform_test):
    if cfg.dataset=='cifar10':
        trainset = CIFAR10(cfg.dataloader['rootdir'],
                           download=cfg.dataloader['download'],
                           train=True,
                           transform=transform_train)
        testset  = CIFAR10(cfg.dataloader['rootdir'],
                           download=cfg.dataloader['download'],
                           train=False,
                           transform=transform_test)
    else:
        trainset = CIFAR100(cfg.dataloader['rootdir'],
                            download=cfg.dataloader['download'],
                            train=True,
                            transform=transform_train)
        testset  = CIFAR100(cfg.dataloader['rootdir'],
                            download=cfg.dataloader['download'],
                            train=False,
                            transform=transform_test)
    return trainset, testset


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                    DATALOADERS                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def get_dataloaders():
    transform_train, transform_test = get_transforms()
    trainset, testset = get_datasets(transform_train, transform_test)
    trainloader = DataLoader(trainset,
                             batch_size=cfg.dataloader['train_batch_size'],
                             shuffle=True,
                             num_workers=cfg.dataloader['nb_workers'])
    testloader  = DataLoader(testset, 
                             batch_size=cfg.dataloader['test_batch_size'], 
                             shuffle=False,
                             num_workers=cfg.dataloader['nb_workers'],)
    return trainloader, testloader






