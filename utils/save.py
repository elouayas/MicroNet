import os
import torch


def save(dataset, name, model):
    if not os.path.isdir('./checkpoints/cifar100/'):
        os.mkdir('./checkpoints/cifar100/')
    if not os.path.isdir('./checkpoints/cifar10/'):
        os.mkdir('./checkpoints/cifar10/')
    checkpoints_path = './checkpoints/' + dataset + '/'
    filename = 'INTERRUPTED_' + name + '.pt'
    torch.save(model.net.state_dict(), checkpoints_path + filename)
    print()
    print(80*'_')
    print('Training Interrupted')
    print('Current State saved.')