import os
from time import time


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                         TIMER                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #  

def timed(function):
    def wrapper(*args, **kwargs):
        start = time()
        output = function(*args, **kwargs)
        end = time()
        run_time = end-start
        h, m, s = run_time//3600, (run_time%3600)//60, run_time%60
        print('Function mesured...: ' + str(function))
        print('Time taken.........: %dh %dm %.2fs' % (h, m, s))
        return output
    return wrapper


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                         SAVE                                        | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #  

def saveWrapper(function):
    def wrapper(*args, **kwargs):
        if not os.path.isdir('../checkpoints/'):
            os.mkdir('../checkpoints/')    
        if not os.path.isdir('../checkpoints/cifar100/'):
            os.mkdir('../checkpoints/cifar100/')
        if not os.path.isdir('../checkpoints/cifar10/'):
            os.mkdir('../checkpoints/cifar10/')
        interrupted = args[1]
        output = function(*args, **kwargs)
        if interrupted:
            print()
            print(80*'_')
            print('Training Interrupted')
            print('Current State saved.')
        return output
    return wrapper


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                         SUMMARY                                     | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def summary(cfg):
    def summary_decorator(function):
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            print(80*'_')
            print('Training settings  : \n')
            print(f'Dataset...................:  {cfg.dataset}')
            print(f"Net.......................:  {cfg.model['net']}")
            print(f"Optimizer.................:  {cfg.optim['type']}")
            print(f"LookAhead.................:  {str(cfg.optim['use_lookahead'])}")
            print(f"Learning Rate Scheduler...:  {cfg.scheduler['type']}")
            print(f"Decay.....................:  {str(cfg.scheduler['use_delay'])}")
            print(f"Label Smoothing...........:  {str(cfg.model['label_smoothing'])}")
            print(f"CutMix....................:  {str(cfg.train['use_cutmix'])}")
            print(f"Number of epochs..........:  {str(cfg.train['nb_epochs'])}")
            #print(f"Use Binary Connect........:  {str(train_config['use_binary_connect'])}")
            #print(f"Use Soft Pruning..........:  {str(train_config['use_pruning'])}")
            print(80*'_')
            return output
        return wrapper
    return summary_decorator
    

# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                        VERBOSE                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ # 

def verbose(function):
    """ This verbose decorator is specific to the one_epoch_step() method
        of the Trainer class.
    """ 
    def wrapper_train(*args, **kwargs):
        current_epoch, nb_epochs = args[1], args[2]
        print(80*'_')
        print('EPOCH %d / %d' % (current_epoch+1, nb_epochs))
        _, train_loss, train_acc, test_loss, test_acc, lr = function(*args, **kwargs)
        print()
        print('Train Loss................: {:.2f}'.format(train_loss))
        print('Test Loss.................: {:.2f}'.format(test_loss))
        print('Train Accuracy............: {:.2f}'.format(train_acc))
        print('Test Accuracy.............: {:.2f}'.format(test_acc))
        print()
        print('Current Learning Rate.....: {:.10f}'.format(lr))
        if test_acc > wrapper_train.best_acc:
            wrapper_train.best_acc = test_acc
        print('Best Test Accuracy........: {:.2f}'.format(wrapper_train.best_acc))
        return _, train_loss, train_acc, test_loss, test_acc, lr
    wrapper_train.best_acc = 0
    return wrapper_train


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                     TO TENSORBOARD                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ # 

def toTensorboard(function):
    """ This verbose decorator is specific to the one_epoch_step() method
        of the Trainer class.
    """ 
    def wrapper_train(*args, **kwargs):
        epoch = args[1]
        writer, train_loss, train_acc, test_loss, test_acc, lr = function(*args, **kwargs)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Learning Rate/lr', lr, epoch)
        return writer, train_loss, train_acc, test_loss, test_acc, lr
    return wrapper_train