"""Generic Configuration File For Training Pipeline on CIFAR10/100"""

from brevitas.core.quant import QuantType


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                      DATASET CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
DATASET CONFIG

Args:

    dataset (str):  'cifar10' or 'cifar100'.
                    This will affect:
                        * the dataloaders in dataloader.py
                        * the final layer of model (via the num_classes param)
"""

dataset = 'cifar100'



# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                      MODEL CONFIG                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
MODEL CONFIG

In all the code we call a model the set of:
    * a network
    * an optimizer
    * a scheduler
    * a criterion (a loss)
This is somewhat unusual and one must remind the definition of model and net here.

Args:

    net (str):  defines the network to train
                can be one of: 
                    * resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
                    * densenet100, densenet121, densenet161, densenet169, densenet201, 
                    * efficientnet-b0, ..., efficientnet_b7
                    * wide_resnet_28_10,
                    * pyramidnet272
                one can add new models by:
                    1. adding them to the models/ folder
                    2. modify accordingly the _load_net() method of the Model class (model.py)
    
    mode (str): 'basic' or 'alternative'.
                Refers to a set (optimizer + scheduler) 
                * basic means SGD + ROP
                  SGD params: lr = 01, momentum = 0.9, nesterov = True, weight_decay = 5e-4
                  ROP params: mode = 'min', factor = 0.2, patience = 20, verbose = True
                * alternative means RangerLars + DelayedCosineAnnealing
                  Be carefull: when using DelayedCosineAnnealing, 
                  params depending on the number of training epochs are calculated.
                  Thus one must avoid setting an arbitrary high number of epochs
                  with early stopping but must instead set a "True" number of epochs". 

    
    label_smoothing (bool): True or False.
                            If True, use utils.LabelSmoothingCrossEntropy 
                            instead of vanilla CrossEntropy.
    
    smoothing (float):  If label_smoothing is False, this has no effect.
                        If label_smoothing is True, and smoothing = x, 
                        the one true label will be 1-x instead of x, 
                        and others false labels will be x instead of 0. 
    
    reduction (str):    'mean' or 'sum.
                        If label_smoothing is False, this has no effect.
                        The smoothed cross entropy (sce) is in general:
                            * sce(i) = (1-eps)ce(i) + eps*reduced_loss
                            where eps is the smoothing param
                        If reduction='mean', reduced_loss = sum(ce(j))/N
                        If reduction='sum',  reduced_loss = sum(ce(j))    
    
    quantize (bool):    True or False
                        Use a quantized network for training or not.
                        If True, net MUST be one of resnet or wide_resnet

    weight_quant_type (QuantType):  BINARY, TERNARY, INT, or FP.
                                    What precision to quantize weights. 
"""



model = {
    'net': 'resnet20',
    'mode': 'basic',
    'label_smoothing': False,
    'smoothing': 0.1,
    'reduction': 'mean', 
    'quantize': False,
    'weight_quant_type': QuantType.INT,
    'weight_bit_width': 8
}



# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                   DATALOADER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
DATALOADER CONFIG

Note that the default data augmentation parameters can be modified in dataloaders.py
via the aug variable in get_transforms().
Cutout parameters should be modified only in the following dictionary.

Args:

    rootdir (str):  path to the dataset

    download (bool): Download data if not found in rootdir. 

    train_batch_size (int): Batch size for the training dataloader.

    test_batch_size (int):  Batch size for the testing dataloader.

    nb_workers (int): number of threads to load data. 
                      Param for the vanilla Pytorch Dataloader class.

    data_aug (bool): control the use of data augmentation.
                     Dataloaders will always use a minimal augmentation:
                        * Random crop + Random Horizontal Flip
                     If True, CIFARPolicy from AutoAugment will be added

    fast_aug (bool): If True, Fast Auto Augment will be used instead of Auto Augment. 
    
    use_cutout (bool): control the use of Cutout in addition to standart data augmentation.

    n_holes (int): If use_cutout is False, this has no effect.
                   Else, set the number of holes to cut out.

    length (int): If use_cutout is False, this has no effect.
                  In our Cutout implementation, holes are squares. 
                  One can set here the side length of those squares.

    resize (bool): When using certain big network, the small CIFAR images (32*32)
                   need to be upsampled (for instance when training and efficientnet).
                   If True, CIFAR images will be resized to 224*224 (standart imageNet size). 
                   The upsampling is performed using bicubic interpolation from PIL.Image.
"""

dataloader = {
    'rootdir': './data/',
    'download': True,
    'train_batch_size': 32,
    'test_batch_size': 16,
    'nb_workers': 6,
    'data_aug': True,
    'fast_aug': False,
    'use_cutout': False,
    'n_holes': 1,
    'length': 16,
    'resize': False,
    'use_fastaugm': False
}

# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                     PRUNING CONFIG                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
PRUNING CONFIG:

NOT FONCTIONAL FOR NOW. DO NOT USE PRUNING
"""

pruning = {
    'pruning_rate': 0.6, # for non asymptotic pruning only
    'asymptotic': True,
    'min': 0, # minimal pruning rate for asymptotic pruning
    'goal': 0.8, # goal pruning rate for asymptotic pruning
    'D': 1/8, # asymptotic pruning rate speed: at D*nb_epochs, pruning rate will be (3/4)*goal.
    # verbose can be:
    # 0 for no verbose
    # 1 for soft verbose (total zeros and nonzeros)
    # 2 for hard verbose (zeros and nonzeros counts per layer)
    'verbose': 1,
}


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                    TRAIN CONFIG                                     | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
TRAIN CONFIG

Args:

    nb_epochs (int): number of training epoch. 
                     For each training epoch, one testing step is performed.
    
    use_early_stopping (bool): If True, the training will stop automatically when 
                               the loss stagnate for a certain amount of epochs.

    patience (int): If use_early_stopping is False, this has no effect.
                    Else, this controls the number of epochs to wait with no loss decrease
                    before stopping the training.

    delta (float): If use_early_stopping is False, this has no effect.
                   Else, this controls the decrease of loss to stop the early stopping
                   (or, to be precise, to reset its patience).

    use_cutmix (bool): controls the use of cutmix.
                       If True, Trainer.train() will call the Cutmix class 
                       from utils.augment.cutmix with a probability p.

    beta (float): If use_cutmix is False, this has no effect.
                  Else, a number is generated via numpy.random.beta(beta,beta)

    p (float): If use_cutmix is False, this has no effect.
               Else, Trainer.train() will call Cutmix with this probability. 

    use_binary_connect (bool): Controls the binarization of the network weights.

    use_pruning (bool): Controls the pruning of the network filters.

    distillation (bool): Controls the "training mode", either a standart train or 
                         or student-teacher train, with distillation params specified in 
                         teacher_params.
""" 

train = {
    'nb_epochs' : 1,
    'use_early_stopping': True,
    'patience': 50,
    'delta': 0.01,
    'use_cutmix': False,
    'beta': 1.0,
    'p': 0.5,
    'use_binary_connect': False,
    'use_pruning':False,
    'verbose': True,
    'distillation':False
}


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                 DISTILLATION CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
DISTILLATION CONFIG

NOT FULLY IMPLEMENTED YET.

"""


# CHANGER TEACHER CONFIG, POUR UTILISER LA CLASSE MODEL DANS LE TRAINER
#'teacher_path':'model_best.pth.tar', pyram200
#'cifar100_pyramid272_top1_11.74.pth' pyram272


teacher  = {
    'teacher_path':'checkpoints/model_best.pth.tar',
    'lambda_hkd':10,
    'lambda_gkd':0,
    'lambda_rkd':0,
    'pool3_only':False,
    'temp':4,
    'power':1,
    'k':128,
    'intra_only':False,
    'inter_only':False,
    'net': 'pyramidnet200',
}



# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                        LOG CONFIG                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #


def get_experiment_name():
    basename= model['net'] + '_' + model['mode']
    if model['quantize']:
        basename += '_quant'
    if model['label_smoothing']:
        basename += '_ls'
    if dataloader['data_aug']:
        if dataloader['fast_aug']:
            basename += '_faa'
        else:
            basename += '_aa'
    if dataloader['use_cutout']:
        basename += '_cutout'
    if dataloader['resize']:
        basename += '_resized'
    if train['use_pruning']:
        basename += '_pruned'
    if train['use_binary_connect']:
        basename += 'bc'
    if train['distillation']:
        basename += 'student_fromTeacher_'+teacher['net']
    return basename



"""
LOG CONFIG

This should never be modified as it is generated with regards to all other config params.
"""    

log = {
    'tensorboard_path': './runs/'+dataset+'/'+get_experiment_name(),
    'checkpoints_path': './checkpoints/'+dataset+'/'+get_experiment_name()+'.pt'
}



    