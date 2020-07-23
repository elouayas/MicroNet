""" Decorators used by the LightningModel and the Trainer classes
    Mostly logging and displaying handlers
"""

import os
from dataclasses import dataclass
from tqdm import tqdm
from pytorch_lightning import Callback

# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                     FANCY DISPLAY                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class FancyDisplay():

    current_loss:        str = '| Current training Loss......:'
    current_acc:         str = '| Current training Accuracy..:'
    current_lr:          str = '| Current Learning Rate......:'

    last_avg_train_loss: str = '| Training loss..............:'
    last_avg_val_loss:   str = '| Validation loss............:'
    last_avg_train_acc:  str = '| Training accuracy..........:'
    last_avg_val_acc:    str = '| Validation accuracy........:'

    best_avg_train_loss: str = '| Training Loss..............:'
    best_avg_val_loss:   str = '| Validation Loss............:'
    best_avg_train_acc:  str = '| Training Accuracy..........:'
    best_avg_val_acc:    str = '| Validation Accuracy........:'




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         STATE                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class State():

    def __init__(self):
        self.current_loss        = 9.9999
        self.current_acc         = 0.
        self.current_lr          = 0.
        self.last_avg_train_loss = 9.999
        self.last_avg_val_loss   = 9.9999
        self.last_avg_train_acc  = 0.
        self.last_avg_val_acc    = 0.
        self.best_avg_train_loss = 9.9999
        self.best_avg_val_loss   = 9.9999
        self.best_avg_train_acc  = 0.
        self.best_avg_val_acc    = 0.
        self.table              = Table()

    def update_current_train(self, output):
        self.current_loss = output['loss']
        self.current_acc  = output['acc']
        self.current_lr   = output['lr']
        self.table.update_current(self.current_loss, self.current_acc, self.current_lr)

    def update_best_average(self):
        self.best_avg_train_loss = min(self.best_avg_train_loss, self.last_avg_train_loss)
        self.best_avg_val_loss   = min(self.best_avg_val_loss,   self.last_avg_val_loss)
        self.best_avg_train_acc  = max(self.best_avg_train_acc,  self.last_avg_train_acc)
        self.best_avg_val_acc    = max(self.best_avg_val_acc,    self.last_avg_val_acc)
        self.table.update_best_average(self.best_avg_train_loss, self.best_avg_val_loss,
                                       self.best_avg_train_acc, self.best_avg_val_acc)

    def update_last_average(self, output):
        self.last_avg_train_loss = output['loss']
        self.last_avg_val_loss   = output['val_loss']
        self.last_avg_train_acc  = output['acc']
        self.last_avg_val_acc    = output['val_acc']
        self.table.update_last_average(self.last_avg_train_loss, self.last_avg_val_loss,
                                       self.last_avg_train_acc,  self.last_avg_val_acc)




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    TQDM DESCRIPTOR                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #


class Descriptor:

    def __init__(self, position):
        self.bar = tqdm(total=0, position=position, bar_format='{desc}')




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          TABLE                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

#TODO: Make tqdm descriptor a class with a method to update string 
#      (and another to update position ?)

class Table:
    """
        Table to be display in terminal, showing current training infos.
        Updated every batch.
    """
    def __init__(self):
        self.strings = FancyDisplay()
        # titles
        self.top_title = self._init_title( 3,  4,  5)
        self.mid_title = self._init_title( 9, 10, 11)
        self.bot_title = self._init_title(16, 17, 18)
        self.last_line = self._init_descriptor(23)
        # current epoch
        self.current = {'loss': self._init_descriptor(6),
                        'acc':  self._init_descriptor(7),
                        'lr':   self._init_descriptor(8)}
        # last epoch average
        self.last_epoch_avg = {'train_loss': self._init_descriptor(12),
                               'val_loss':   self._init_descriptor(13),
                               'train_acc':  self._init_descriptor(14),
                               'val_acc':    self._init_descriptor(15)}
        # all training best
        self.best  = {'train_loss': self._init_descriptor(19),
                      'val_loss':   self._init_descriptor(20),
                      'train_acc':  self._init_descriptor(21),
                      'val_acc':    self._init_descriptor(22)}
        self.last_line = self._init_descriptor(23)

    @staticmethod
    def _init_descriptor(pos):
        return tqdm(total=0, position=pos, bar_format='{desc}')

    def _init_title(self, pos1, pos2, pos3):
        return [self._init_descriptor(pos1),
                self._init_descriptor(pos2),
                self._init_descriptor(pos3)]

    @staticmethod
    def _set_title(title, string, offset):
        title[0].set_description_str('+' + 40*'-' + '+')
        title[1].set_description_str('| ' + string + offset*' ' + '|')
        title[2].set_description_str('+' + 40*'-' + '+')

    def _update(self, descriptor, string, value):
        descriptor.set_description_str(f'{string} {value : 2f} |')

    def update_current(self, loss, acc, lr):
        self._set_title(self.top_title, 'CURRENT EPOCH', 26)
        self._update(self.current['loss'], self.strings.current_loss, loss)
        self._update(self.current['acc'],  self.strings.current_acc,  acc)
        self._update(self.current['lr'],   self.strings.current_lr,   lr)

    def update_last_average(self, loss, val_loss, acc, val_acc):
        self._set_title(self.mid_title, 'LAST EPOCH (average)', 19)
        self._update(self.last_epoch_avg['train_loss'], self.strings.last_avg_train_loss, loss)
        self._update(self.last_epoch_avg['val_loss'],   self.strings.last_avg_val_loss,   val_loss)
        self._update(self.last_epoch_avg['train_acc'],  self.strings.last_avg_train_acc,  acc)
        self._update(self.last_epoch_avg['val_acc'],    self.strings.last_avg_val_acc,    val_acc)

    def update_best_average(self, train_loss, val_loss, train_acc, val_acc):
        self._set_title(self.bot_title, 'BEST SO FAR (one epoch average)', 8)
        self._update(self.best['train_loss'], self.strings.best_avg_train_loss, train_loss)
        self._update(self.best['val_loss'],   self.strings.best_avg_val_loss,   val_loss)
        self._update(self.best['train_acc'],  self.strings.best_avg_train_acc,  train_acc)
        self._update(self.best['val_acc'],    self.strings.best_avg_val_acc,    val_acc)
        self.last_line.set_description_str('+' + 40*'-' + '+')



# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    VERBOSE CALLBACK                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class VerboseCallback(Callback):

    def __init__(self):
        self.state    = State()

    def on_batch_end(self, trainer, pl_module):
        output = trainer.callback_metrics
        if not output:
            return
        self.state.update_current_train(output)

    def on_epoch_end(self, trainer, pl_module):
        output = trainer.callback_metrics
        self.state.update_last_average(output)
        self.state.update_best_average()

    def on_fit_end(self, trainer):
        print(2*'\n')

    def on_train_start(self, trainer, pl_module):
        os.system('cls' if os.name == 'nt' else 'clear')
        