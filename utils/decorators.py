""" Decorators used by the LightningModel and the Trainer classes
    Mostly logging and displaying handlers
"""


import sys
from tqdm import tqdm

def move (y, x):
    print("\033[%d;%dH" % (y, x))

def verbose(function):
    # move cursor
    move_cursor     = tqdm(total=0, position=24,  bar_format='{desc}')
    # fancy display
    top_title_top   = tqdm(total=0, position=3,  bar_format='{desc}')
    top_title_mid   = tqdm(total=0, position=4,  bar_format='{desc}')
    top_title_bot   = tqdm(total=0, position=5,  bar_format='{desc}')
    # all training best
    best_train_loss = tqdm(total=0, position=6,  bar_format='{desc}')
    best_val_loss   = tqdm(total=0, position=7,  bar_format='{desc}')
    best_train_acc  = tqdm(total=0, position=8,  bar_format='{desc}')
    best_val_acc    = tqdm(total=0, position=9,  bar_format='{desc}')
    # fancy display
    mid_title_top   = tqdm(total=0, position=10,  bar_format='{desc}')
    mid_title_mid   = tqdm(total=0, position=11,  bar_format='{desc}')
    mid_title_bot   = tqdm(total=0, position=12,  bar_format='{desc}')
    # current epoch
    loss_status     = tqdm(total=0, position=13, bar_format='{desc}')
    acc_status      = tqdm(total=0, position=14, bar_format='{desc}')
    lr_status       = tqdm(total=0, position=15, bar_format='{desc}')
    # fancy display
    bot_title_top   = tqdm(total=0, position=16, bar_format='{desc}')
    bot_title_mid   = tqdm(total=0, position=17, bar_format='{desc}')
    bot_title_bot   = tqdm(total=0, position=18, bar_format='{desc}')
    # last epoch average
    avg_train_loss  = tqdm(total=0, position=19, bar_format='{desc}')
    avg_val_loss    = tqdm(total=0, position=20, bar_format='{desc}')
    avg_train_acc   = tqdm(total=0, position=21, bar_format='{desc}')
    avg_val_acc     = tqdm(total=0, position=22, bar_format='{desc}')
    # last table line
    bot_table_line  = tqdm(total=0, position=23, bar_format='{desc}')
    def wrapper(*args, **kwargs):
        output = function(*args, **kwargs)
        if function.__name__ == 'training_epoch_end':
            wrapper.avg_train_loss, wrapper.avg_train_acc = output['loss'], output['log']['acc']
            #wrapper.current_lr = output['log']['lr']
        elif function.__name__ == 'validation_epoch_end':
            wrapper.avg_vall_loss = output['val_loss']
        elif function.__name__ == 'validation_step':
            loss = output['val_loss']
            if loss < wrapper.best_val_loss:
                wrapper.best_val_loss = loss
        elif function.__name__ == 'training_step':
            loss, acc = output['loss'], output['acc']
            wrapper.current_lr = output['lr']
            if acc > wrapper.best_train_acc:
                wrapper.best_train_acc = acc
            if loss < wrapper.best_train_loss:
                wrapper.best_train_loss = loss
            # best so far
            top_title_top.set_description_str('+' + 54*'-' + '+')
            top_title_mid.set_description_str('| ' + 'BEST SO FAR' + 42*' ' + '|')
            top_title_bot.set_description_str('+' + 54*'-' + '+')
            best_train_loss.set_description_str(
                f'| Best Training Loss.....................: {wrapper.best_train_loss : .4f}' + 5*' ' + '|')
            best_val_loss.set_description_str(
                f'| Best Validation Loss...................: {  wrapper.best_val_loss : .4f}' + 5*' ' + '|')
            best_train_acc.set_description_str(
                f'| Best Training Accuracy.................: { wrapper.best_train_acc : .4f}' + 5*' ' + '|')
            best_val_acc.set_description_str(
                f'| Best Validation Accuracy...............: {   wrapper.best_val_acc : .4f}' + 5*' ' + '|')
            # current epoch
            mid_title_top.set_description_str('+' + 54*'-' + '+')
            mid_title_mid.set_description_str('| ' + 'CURRENT EPOCH' + 40*' ' + '|')
            mid_title_bot.set_description_str('+' + 54*'-' + '+')
            loss_status.set_description_str(
                f'| Current training Loss..................: {loss : .4f}' + 5*' ' + '|')
            acc_status.set_description_str(
                f'| Current training Accuracy..............: { acc : .4f}' + 5*' ' + '|')
            lr_status.set_description_str(
                f'| Current Learning Rate..................: { wrapper.current_lr : .4f}' + 5*' ' + '|')
            # last epoch average
            bot_title_top.set_description_str('+' + 54*'-' + '+')
            bot_title_mid.set_description_str('| ' + 'LAST EPOCH AVERAGE' + 35*' ' + '|')
            bot_title_bot.set_description_str('+' + 54*'-' + '+')
            avg_train_loss.set_description_str(
                f'| Average training loss last epoch.......: {wrapper.avg_train_loss : .4f}' + 5*' ' + '|')
            avg_val_loss.set_description_str(
                f'| Average validation loss last epoch.....: {  wrapper.avg_val_loss : .4f}' + 5*' ' + '|')
            avg_train_acc.set_description_str(
                f'| Average training accuracy last epoch...: { wrapper.avg_train_acc : .4f}' + 5*' ' + '|')
            avg_val_acc.set_description_str(
                f'| Average validation accuracy last epoch.: {   wrapper.avg_val_acc : .4f}' + 5*' ' + '|')
            # last line
            bot_table_line.set_description_str('+' + 54*'-' + '+')
            move_cursor.set_description_str("\033[0;0H")
            sys.stdout.flush()
        return output
    wrapper.best_train_acc,  wrapper.best_val_acc  = 0, 0
    wrapper.best_train_loss, wrapper.best_val_loss = 9, 9
    wrapper.avg_train_loss,  wrapper.avg_val_loss  = 0, 0
    wrapper.avg_train_acc,   wrapper.avg_val_acc   = 0, 0
    wrapper.current_lr = 0.0001
    return wrapper
