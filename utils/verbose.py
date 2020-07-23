""" Decorators used by the LightningModel and the Trainer classes
    Mostly logging and displaying handlers
"""

import os
from tqdm import tqdm
from pytorch_lightning import Callback

# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                     FANCY DISPLAY                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class FancyDisplay():

    def __init__(self, width):
        self.current_loss        = '| Current training Loss......:'
        self.current_acc         = '| Current training Accuracy..:'
        self.current_lr          = '| Current Learning Rate......:'
        self.last_avg_train_loss = '| Training loss..............:'
        self.last_avg_val_loss   = '| Validation loss............:'
        self.last_avg_train_acc  = '| Training accuracy..........:'
        self.last_avg_val_acc    = '| Validation accuracy........:'
        self.best_avg_train_loss = '| Training Loss..............:'
        self.best_avg_val_loss   = '| Validation Loss............:'
        self.best_avg_train_acc  = '| Training Accuracy..........:'
        self.best_avg_val_acc    = '| Validation Accuracy........:'
        self.title_border        = '+' + (width-2)*'-' + '+'




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    TQDM DESCRIPTOR                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Descriptor:

    def __init__(self, position, string):
        self.tqdm   = tqdm(total=0, position=position, bar_format='{desc}')
        self.string = string

    def update(self, value):
        self.tqdm.set_description_str(f'{self.string} {value : 2f} |')

    def set_title(self):
        self.tqdm.set_description_str(self.string)




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
        self.strings   = FancyDisplay(width)
        # titles
        self.top_title = self._init_title( 3,  4,  5, 'CURRENT EPOCH')
        self.mid_title = self._init_title( 9, 10, 11, 'LAST EPOCH (average)')
        self.bot_title = self._init_title(16, 17, 18, 'BEST SO FAR (one epoch average)')
        self.last_line = Descriptor(23, self.strings.title_border)
        # current epoch
        self.current = {'loss': Descriptor(6, self.strings.current_loss),
                         'acc': Descriptor(7, self.strings.current_acc),
                          'lr': Descriptor(8, self.strings.current_lr)}
        # last epoch average
        self.last_epoch_avg = {'train_loss': Descriptor(12, self.strings.last_avg_train_loss),
                                 'val_loss': Descriptor(13, self.strings.last_avg_val_loss),
                                'train_acc': Descriptor(14, self.strings.last_avg_train_acc),
                                  'val_acc': Descriptor(15, self.strings.last_avg_val_acc)}
        # all training best
        self.best  = {'train_loss': Descriptor(19, self.strings.best_avg_train_loss),
                        'val_loss': Descriptor(20, self.strings.best_avg_val_loss),
                       'train_acc': Descriptor(21, self.strings.best_avg_train_acc),
                         'val_acc': Descriptor(22, self.strings.best_avg_val_acc)}

    def _init_title(self, pos1, pos2, pos3, string):
        offset = self.width - 3 - len(string)
        return [Descriptor(pos1, self.strings.title_border),
                Descriptor(pos2, '| ' + string + offset*' ' + '|'),
                Descriptor(pos3, self.strings.title_border)]

    def _set_title(self, title):
        title[0].set_title()
        title[1].set_title()
        title[2].set_title()

    def update_current(self, loss, acc, lr):
        self._set_title(self.top_title)
        self.current['loss'].update(loss)
        self.current['acc'].update(acc)
        self.current['lr'].update(lr)

    def update_last_average(self, loss, val_loss, acc, val_acc):
        self._set_title(self.mid_title)
        self.last_epoch_avg['train_loss'].update(loss)
        self.last_epoch_avg['val_loss'].update(val_loss)
        self.last_epoch_avg['train_acc'].update(acc)
        self.last_epoch_avg['val_acc'].update(val_acc)

    def update_best_average(self, train_loss, val_loss, train_acc, val_acc):
        self._set_title(self.bot_title)
        self.best['train_loss'].update(train_loss)
        self.best['val_loss'].update(val_loss)
        self.best['train_acc'].update(train_acc)
        self.best['val_acc'].update(val_acc)
        self.last_line.set_title()




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
        