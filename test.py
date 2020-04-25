import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import KMeans
import k_means

from model import Model
from trainer import Trainer
import config as cfg

from utils import save

c=2
k=40 #c'est pas le CR
logk=6
CR=0


def build():
    print('Building Model...')
    model = Model(cfg.model['net'])
    trainer = Trainer(model)
    print(trainer)
    return model, trainer


model, trainer = build()
#model.load('checkpoints/cifar100/resnet20_basic.pt')
print(model.num_params)


kmeans=k_means.K_means(model.net)
CR=kmeans.calcul_rapport(c,k,logk)
print("CR="+str(CR))
kmeans.quantization(c,k)
print("kmeans done")


trainer.run()


save(cfg.dataset, 'kmean_test_resnet20', model)
