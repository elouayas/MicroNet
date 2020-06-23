from utils.dataset import CIFAR100
from torch.utils.data import DataLoader

def get_dataloaders(config):
    trainset = CIFAR100(config.dataloader['rootdir'], train = True,  scale = 255)
    testset  = CIFAR100(config.dataloader['rootdir'], train = False, scale = 255)
    trainloader = DataLoader(trainset,
                             batch_size  = config.dataloader['train_batch_size'],
                             shuffle     = True,
                             num_workers = config.dataloader['nb_workers'])
    testloader  = DataLoader(testset, 
                             batch_size  = config.dataloader['test_batch_size'], 
                             shuffle     = False,
                             num_workers = config.dataloader['nb_workers'],)
    return trainloader, testloader






