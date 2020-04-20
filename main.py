"""
+---------------------------------------------------------------------------------------+
|                                                                                       |
|                                         MAIN                                          |
|                                                                                       |
+---------------------------------------------------------------------------------------+
"""

import torch
from model import Model
from trainer import Trainer
from config import dataset, model_config, dataloader_config, train_config

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


def train():
    print('Building Model...')
    model = Model(model_config, dataloader_config, dataset)
    trainer = Trainer(model, dataloader_config, train_config)
    print(trainer)
    try:
        trainer.run()
    except KeyboardInterrupt:
        net = model.net.summary['net']
        optimizer =  model.net.summary['optimizer']
        scheduler = model.net.summary['scheduler']
        basename = net + '_' + optimizer + '_' + scheduler + '.pt'
        if dataloader_config['pretrained']:
            filename = 'interrupted_' + 'pretrained_' + basename
        else:
            filename = 'interrupted_' + basename
        path = train_config['checkpoints_path'] + filename
        torch.save(model.net.state_dict(), path)
        print()
        print(80*'_')
        print('Training Interrupted')
        print('Current State saved.')


def test():
    print('Building Model...')
    model = Model(model_config, dataloader_config, dataset)
    trainer = Trainer(model, dataloader_config, train_config)
    model = torch.load('./checkpoints/wide-resnet-28x10.t7')
    model.eval()
    test_loss, test_acc = trainer.test()
    print(80*'_')
    print(f'Loss......: {test_loss}')
    print(f'Accuracy..: {test_acc}')
    print(80*'_')


def load():
    print('Building Model...')
    model = Model(model_config, dataloader_config, dataset)
    trainer = Trainer(model, dataloader_config, train_config)
    print(trainer)


if __name__ == '__main__':
    #load()
    train()
    # test()
