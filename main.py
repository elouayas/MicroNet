""" Main Python file to start training """

import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from model import LightningModel
from utils.verbose import VerboseCallback
import config as cfg

def make_config():
    """
    Uses the 7 dataclasses from config.py to instanciate a Config meta dataclass.
    this config meta dataclass can then be used to instanciate a model.
    """
    dataset      = cfg.Dataset()
    dataloader   = cfg.Dataloader()
    model        = cfg.Model()
    optim        = cfg.Optimizer()
    scheduler    = cfg.Scheduler()
    train        = cfg.Train()
    distillation = cfg.Distillation()
    return cfg.Config(dataset, dataloader, model, optim, scheduler, train, distillation)

def init_model(config):
    """ config must be an instance of the Config dataclass from config.py """
    return  LightningModel(config)


def init_trainer():
    """ Init a Lightning Trainer using from_argparse_args
    Thus every CLI command (--gpus, distributed_backend, ...) become available.
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()
    lr_logger = LearningRateLogger()
    verbose   = VerboseCallback()
    return Trainer.from_argparse_args(args, callbacks = [lr_logger, verbose])


def run_training():
    """ Instanciate a model and a trainer and run trainer.fit(model) """
    config = make_config()
    model, trainer = init_model(config), init_trainer()
    trainer.fit(model)


def test(path):
    model = LightningModel.load_from_checkpoint(path)
    trainer = init_trainer()
    trainer.test(model)


if __name__ == '__main__':
    #run_training()
    test('lightning_logs/version_0/checkpoints/epoch=306.ckpt')
