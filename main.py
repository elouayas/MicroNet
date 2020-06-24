from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from model import Model
import config

model = Model(config)

parser = ArgumentParser()
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

lr_logger = LearningRateLogger()

trainer = Trainer.from_argparse_args(args, callbacks = [lr_logger])
trainer.fit(model)  