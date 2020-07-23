""" Decorators used by the LightningModel and the Trainer classes
    Mostly logging and displaying handlers
"""

import os
from tqdm import tqdm
from dataclasses import dataclass
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
# |                                    TQDM DESCRIPTOR                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Descriptor:

    def __init__(self, width, position, string):
        self.tqdm   = tqdm(total=0, position=position, bar_format='{desc}')
        self.string = string
        self.offset = width - 2 - 8 - len(string)

    def update(self, value):
        status = self.string + ' {:4f}'.format(value) + self.offset*' ' + '|'
        self.tqdm.set_description_str(status)

# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                       TQDM TITLE                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Title:

    def __init__(self, position, width, string=None, top_border_only=False):
        self.top_border_only = top_border_only
        self.top_border = tqdm(total=0, position=position-1, bar_format='{desc}')
        if not self.top_border_only:
            self.title      = tqdm(total=0, position=position,   bar_format='{desc}')
            self.bot_border = tqdm(total=0, position=position+1, bar_format='{desc}')
            self.string     = '|' + string + (width-2-len(string))*' ' + '|'
        self.str_border = '+' + (width-2)*'-' + '+'

    def display(self):
        self.top_border.set_description_str(self.str_border)
        if not self.top_border_only:
            self.title.set_description_str(self.string)
            self.bot_border.set_description_str(self.str_border)




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
    def __init__(self, width=42):
        self.width     = width
        self.strings   = FancyDisplay()
        # titles
        self.top_title = Title(4,  width, 'CURRENT EPOCH')
        self.mid_title = Title(10, width, 'LAST EPOCH (average)')
        self.bot_title = Title(17, width, 'BEST SO FAR (one epoch average)')
        self.last_line = Title(24, width, top_border_only=True)
        # current epoch
        self.current = {'loss': Descriptor(width, 6, self.strings.current_loss),
                         'acc': Descriptor(width, 7, self.strings.current_acc),
                          'lr': Descriptor(width, 8, self.strings.current_lr)}
        # last epoch average
        self.last_epoch_avg = {'train_loss': Descriptor(width, 12, self.strings.last_avg_train_loss),
                                 'val_loss': Descriptor(width, 13, self.strings.last_avg_val_loss),
                                'train_acc': Descriptor(width, 14, self.strings.last_avg_train_acc),
                                  'val_acc': Descriptor(width, 15, self.strings.last_avg_val_acc)}
        # all training best
        self.best  = {'train_loss': Descriptor(width, 19, self.strings.best_avg_train_loss),
                        'val_loss': Descriptor(width, 20, self.strings.best_avg_val_loss),
                       'train_acc': Descriptor(width, 21, self.strings.best_avg_train_acc),
                         'val_acc': Descriptor(width, 22, self.strings.best_avg_val_acc)}

    def update_current(self, loss, acc, lr):
        self.top_title.display()
        self.current['loss'].update(loss)
        self.current['acc'].update(acc)
        self.current['lr'].update(lr)

    def update_last_average(self, loss, val_loss, acc, val_acc):
        self.mid_title.display()
        self.last_epoch_avg['train_loss'].update(loss)
        self.last_epoch_avg['val_loss'].update(val_loss)
        self.last_epoch_avg['train_acc'].update(acc)
        self.last_epoch_avg['val_acc'].update(val_acc)

    def update_best_average(self, train_loss, val_loss, train_acc, val_acc):
        self.bot_title.display()
        self.best['train_loss'].update(train_loss)
        self.best['val_loss'].update(val_loss)
        self.best['train_acc'].update(train_acc)
        self.best['val_acc'].update(val_acc)
        self.last_line.display()




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         STATE                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class State():

    def __init__(self, table_width=42):
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
        self.table              = Table(table_width)

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
        