import torch
from model import Model
from trainer import Trainer
import config as cfg


def build():
    print('Building Model...')
    model = Model()
    trainer = Trainer(model)
    print(trainer)
    return model, trainer


model, trainer = build()