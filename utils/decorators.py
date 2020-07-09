def verbose(function):
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
