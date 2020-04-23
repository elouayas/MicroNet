# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import os
import torch
from model import Model
from trainer import Trainer
from utils import save
import config as cfg

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

def build():
    print('Building Model...')
    model = Model(cfg.model['net'])
    trainer = Trainer(model)
    print(trainer)
    return model, trainer


def train():
    model, trainer = build()
    try:
        trainer.run()
    except KeyboardInterrupt:
        save(cfg.dataset, cfg.get_experiment_name(), model)


def test(path):
    model, trainer = build()
    model.load(path)
    model.net.eval()
    test_loss, test_acc = trainer.test()
    print(80*'_')
    print(f'Loss......: {test_loss}')
    print(f'Accuracy..: {test_acc}')
    print(80*'_')



if __name__ == '__main__':
    #build()
    train()
    #test()
