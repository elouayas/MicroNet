# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                         TIMER                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #  

from time import time


def timed(function):
    def wrapper(*args, **kwargs):
        start = time()
        output = function(*args, **kwargs)
        end = time()
        run_time = end-start
        print('Function mesured...: ' + str(function))
        print('Time taken.........: %dh %dm %.2fs' % (run_time//3600, (run_time%3600)//60, run_time%60))
        return output
    return wrapper


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                         SUMMARY                                     | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def summary(dataset, model_config, train_config):
    def summary_decorator(function):
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            if model_config['mode'] == 'basic':
                optimizer_name, scheduler_name = 'SGD', 'ROP'
            elif model_config['mode'] == 'alternative':
                optimizer_name, scheduler_name = 'Ranger + LARS', 'Delayed Cosine Annealing'
            print(80*'_')
            print('Training settings  : \n')
            print(f'Dataset...................:  {dataset}')
            print(f"Net.......................:  {model_config['net']}")
            print(f'Optimizer.................:  {optimizer_name}')
            print(f'Learning Rate Scheduler...:  {scheduler_name}')
            print(f"Number of epochs..........:  {str(train_config['nb_epochs'])}")
            print(f"Use Binary Connect........:  {str(train_config['use_binary_connect'])}")
            print(f"Use Soft Pruning..........:  {str(train_config['use_pruning'])}")
            print(80*'_')
            return output
        return wrapper
    return summary_decorator
    
        