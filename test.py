import torch

from sklearn.cluster import KMeans
import k_means

from model import Model
from trainer import Trainer
import config as cfg


def build():
    print('Building Model...')
    model = Model(cfg.model['net'])
    trainer = Trainer(model)
    print(trainer)
    return model, trainer


model, trainer = build()
model.load('checkpoints/cifar100/resnet20_basic.pt')
print(model.num_params)

