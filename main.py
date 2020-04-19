# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                                         MAIN                                          | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
from torch import save 
from model import Model
from trainer import Trainer
from config import dataset, model_config, dataloader_config, train_config, test_config

# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                             THIS FILE SHOULD NOT BE MODIFIED                          | #
# |                   ALL HYPER PARAMETERS SHOULD BE CONFIGURED IN config.py              | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #


def train(dataset, model_config, dataloader_config, train_config):
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
        save(model.net.state_dict(), path)
        print()
        print(80*'_')
        print('Training Interrupted')
        print('Current State saved.')
        
        
def test(dataset, model_config, dataloader_config, train_config, test_config):
    print('Building Model...')
    model = Model(model_config, dataloader_config, dataset)
    model.net.load_state_dict(torch.load(test_config['state_dict_path']))
    trainer = Trainer(model, dataloader_config, train_config)
    test_loss, test_acc = trainer.test()
    print()
    print('Test Loss.................: {:.2f}'.format(test_loss))
    print('Test Accuracy.............: {:.2f}'.format(test_acc))


def load(dataset, model_config, dataloader_config, train_config, test_config):
    print('Building Model...')
    model = Model(model_config, dataloader_config, dataset)
    trainer = Trainer(model, dataloader_config, train_config)
    print(trainer)


if __name__ == '__main__':
    load(dataset, model_config, dataloader_config, train_config, test_config)
    # train(dataset, model_config, dataloader_config, train_config)
    # test(dataset, model_config, dataloader_config, train_config, test_config)