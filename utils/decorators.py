""" Decorators used by the LightningModel and the Trainer classes
    Mostly logging and displaying handlers
"""

def val_verbose(function):
    """ This verbose decorator is specific to the validation_epoch_end method
        of the LightningModel class.
    """
    def wrapper(*args, **kwargs):
        print()
        print()
        print()
        output = function(*args, **kwargs)
        val_loss, val_acc = output['log']['val_loss'], output['log']['val_acc']
        print('Validation Loss.................: {:.2f}'.format(val_loss))
        print('Validation Accuracy.............: {:.2f}'.format(val_acc))
        if val_acc > wrapper.best_acc:
            wrapper.best_acc = val_acc
        print('Best Validation Accuracy........: {:.2f}'.format(wrapper.best_acc))
        print()
        return output
    wrapper.best_acc = 0
    return wrapper

def train_verbose(function):
    """ This verbose decorator is specific to the training_epoch_end method
        of the LightningModel class.
    """
    def wrapper(*args, **kwargs):
        print()
        print()
        print()
        output = function(*args, **kwargs)
        train_loss, train_acc = output['log']['loss'], output['log']['train_acc']
        print('Training Loss.................: {:.2f}'.format(train_loss))
        print('Training Accuracy.............: {:.2f}'.format(train_acc))
        if train_acc > wrapper.best_acc:
            wrapper.best_acc = train_acc
        print('Best Training Accuracy........: {:.2f}'.format(wrapper.best_acc))
        print()
        return output
    wrapper.best_acc = 0
    return wrapper
