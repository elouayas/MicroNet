"""Generic Configuration File For Training Pipeline on CIFAR10/100"""


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
                     Dataloaders will always perform a minimal augmentation:
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
                   need to be upsampled (for instance when training an efficientnet).
                   If True, CIFAR images will be resized to 224*224 (standart imageNet size). 
                   The upsampling is performed using bicubic interpolation from PIL.Image.
                   
    winner_config (bool) : If true, it will use the same config as the winners : auto augment cifar 100 policy, with winner policies
    winner_policies : Data-aug policy of the winners. Set cutout in winner_policies as (0,0) if cutmix or fmix.
"""



dataloader = {
    'rootdir': './data/',
    'download': True,
    'train_batch_size': 32,
    'test_batch_size': 32,
    'nb_workers': 4,
    'data_aug': False,
    'fast_aug': False,
    'use_cutout': False,
    'n_holes': 1,
    'length': 16,
    'resize': False,
    'winner_config':True,
}

winner_policies = [
[("Invert", 0.2, 2)],
[("Contrast", 0.4, 4)],
[("Rotate", 0.5, 1)],
[("TranslateX", 0.4, 3)],
[("Sharpness", 0.5, 3)],
[("ShearY", 0.3, 4)],
[("TranslateY", 0.6, 8)],
[("AutoContrast", 0.6, 3)],
[("Equalize", 0.5, 5)],
[("Solarize", 0.4, 4)],
[("Color", 0.5, 5)],
[("Posterize", 0.2, 2)],
[("Brightness", 0.4, 5)],
[("Cutout", 0.3, 3)],
[("ShearX", 0.1, 3)],
]

"""
Defaut winner_policies config
[[("Invert", 0.2, 2)],[("Contrast", 0.4, 4)],[("Rotate", 0.5, 1)],[("TranslateX", 0.4, 3)],[("Sharpness", 0.5, 3)],[("ShearY", 0.3, 4)],[("TranslateY", 0.6, 8)],
[("AutoContrast", 0.6, 3)],[("Equalize", 0.5, 5)],[("Solarize", 0.4, 4)],[("Color", 0.5, 5)],[("Posterize", 0.2, 2)],[("Brightness", 0.4, 5)],
[("Cutout", 0.3, 3)],[("ShearX", 0.1, 3)],]
"""


# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                    OPTIMIZER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
OPTIMIZER CONFIG

Here one can choose parameters for one given optimizer and also choose to use LookAhead or not.
The optimizer itself params are in differents dicts, one per optimizer.

HOW TO USE:
    - To chose the optimizer itself, use the constant OPTIMIZER
    - To use LookAhead or not, use the constant LOOKAHEAD
    - To change optimizer params, use the appropriate subdict of optim_params

Args:

    type (str): optimizer to use. Can be one of
                * SGD: the old robust optimizer
                * RAlamb: stands for RAdam + LARS
                  RAdam stands for Rectified Adam
    
    params (dict): the optimizer's params. Must be a dict with the params named 
                   exactly as in PyTorch implementation.

    use_lookahead: boolean controlling the use of LookAhead.
                   Using LookAhead with RAlamb usually works well.
                   See RangerLARS.

    lookahead: lookahead params. Must be a dict.

"""


OPTIMIZER = 'SGD'
LOOKAHEAD = False

optim_params = {
    'SGD':    {'lr': 0.1, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 5e-4},
    'RAlamb': {'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay':0}
 }

LookAhead_config = {'alpha': 0.5, 'k': 6}

optim = {
    'type': OPTIMIZER,
    'params' : optim_params[OPTIMIZER],
    'use_lookahead': LOOKAHEAD,
    'lookahead': LookAhead_config
}




# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                    SCHEDULER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
SCHEDULER CONFIG

Here one can choose parameters for one given scheduler and also choose to use Decay or not.
The scheduler itself params are in differents dicts, one per scheduler.

HOW TO USE:
    - To chose the scheduler itself, use the constant SCHEDULER
    - To use Delay or not, use the constant DELAY
    - To change optimizer params, use the appropriate subdict of optim_params

Args:

    type (str): scheduler to use. Can be one of
                * ROP: Reduce LR on Plateau when a given metric stop improving. 
                  Default: reduce when test loss stop decreasing. 
                * MultiStep: classic scheduler: multiply the LR at given milestones
                  by a given gamma.
                * Cosine: Anneals the LR following a decreasing cosine curve.
                  Be careful when using Decay: the arg epochs specifies in how many
                  epochs should the annealing occurs. If for instance the total epochs
                  number is 300 and a decay of 150 is set, the cosine annealing will occurs
                  in the last 150 epochs, thus the epochs params should be set at 150 
                  and not 300.  
                * WarmupCosine: Cosine Annealing but with a warmup at the beginning.
                  The LR will groth during a given number of epoch
                * WarmRestartCosine: Same as WarmupCosine but several warmup phases occur
                  during training instead of only one at the beginning.
    
    params (dict): the scheduler's params. Must be a dict with the params named 
                   exactly as in PyTorch implementation.

    use_decay: boolean controlling the use of Decay.
               If True, the scheduler defined will start being active after the given 
               number of epochs. 
               This is often usefull when using cosine or exponential annealing.

    lookahead: lookahead params. Must be a dict.

"""

SCHEDULER = 'ROP'
DELAY     = False

schedul_params = {
    'ROP'               : {'mode': 'min', 'factor': 0.2, 'patience': 20, 'verbose': True},
    'MultiStep'         : {'milestones': [120, 200], 'gamma': 0.1, 'last_epoch': -1},
    'Cosine'            : {'epochs': 150},
    'WarmupCosine'      : {'base_lr': 0.001, 'target_lr': 0.1, 'warm_up_epoch': 5, 'cur_epoch': 0}, 
    'WarmRestartsCosine': {'T_0': 150, 'T_mult': 1, 'eta_min': 0, 'last_epoch': -1}

}

Delay_config = {'delay_epochs': 150, 'after_scheduler': 'Cosine'}

scheduler = {
    'type': SCHEDULER,
    'params' : schedul_params[SCHEDULER],
    'use_delay': DELAY,
    'delay': Delay_config,
}




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
                    * densenet100, densenet172 
                    * efficientnet-b0, ..., efficientnet_b7
                    * wide_resnet_28_10,
                one can add new models by:
                    1. adding them to the models/ folder
                    2. modify accordingly the _load_net() method of the Model class (model.py)

    activation (str): the activation function to use. Can be one of:
                      * ReLU
                      * Swish
                      * Mish
                      Mish seems to often gives best results.
    
    #TODO: make self_attention available for all networks.
    self_attention (bool): boolean to control the insertion of a simple self attention layer
                           inside the basic blocks of the network. 
                           For now, self attention is available for resnet and densenet only.

    attention_sym (bool): force or not the attention matrix to be symetric.
                          It seems having a symetric help convergence. 
    
    shakedrop (bool) : add shakedrop at the end of the forward,
    
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
    'net': 'densenet100',
    'activation': 'relu',
    'self_attention': False,
    'attention_sym': False,
    'shakedrop':False,
    'label_smoothing': False,
    'smoothing': 0.1,
    'reduction': 'mean', 
    'quantize': False,
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
                       from utils.augment.cutmix with a probability p. (Set cutout in winner_policies as (0,0) if cutmix)

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

Specifies the kind of  distillation to use and its config. 
Note that EfficientNet is not compatible for now with GKD.
For now, it is impossible to train in parallel the teacher and the student.
The teacher must be trained already. 
This will be implemented soon.

Args:

    net (str): specifies the teacher network. Annalogs to model['net']

    teacher_path (str): specifies where to load the weights of the trained teacher network. 

    lambda_x (str): if one of those three lambda is > 0, the associated distillation method
                    will be used. Only one of those lambda should be > 0 and the two others
                    equal to 0. 
                    The resulting loss is of the form NormalLoss() + lambda*DistilLoss() 
    
    temp:   If lambda_hkd = 0, this has no effect.
            When using HKD, the teacher's loss is computed on teacher outputs divided by temp.

    pool3_only: If lambda_gkd = 0, this has no effect.
                When using GKD, the teacher is probed at 3 steps (which we call 3 'pooling').
                If pool3_only=True, the distillation is made only using the third pool, in the 
                very end of the teacher network.

    k:   If lambda_gkd = 0, this has no effect.
         When using GKD, a similarity graph is construced based on the k-nearest neighbor 
         algorithm. This param controls the value of k to construct this graph. 
         This param should be based on the number of params in the network.

    power: If lambda_gkd = 0, this has no effect.
           When usind GKD, this param controls the power of the normalized adjacency matrix
           to use before computing the loss. 
           By considering higher powers of matrices A, we consider higher-order
           geometric relations between inner representations of inputs.

    intra/inter only: If lambda_gkd = 0, this has no effect.
                      When using GKD, intra_only (resp inter_only) controls whereas
                      the similary graph should be constructed using only examples of
                      the same (resp distinct) class, thus focusing on the clustering
                      (resp margin) of classes. 

"""


teacher  = {
    'net': 'densenet172',
    'teacher_path':'checkpoints/cifar100/densenet_172_micronet_basic_wc.pt',
    'lambda_hkd':0,
    'lambda_gkd':10,
    'lambda_rkd':0,
    'temp':4,
    'pool3_only': False,
    'k':128,
    'power':1,
    'intra_only':False,
    'inter_only':False,
}



# +-------------------------------------------------------------------------------------+ # 
# |                                                                                     | #
# |                                        LOG CONFIG                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

"""
LOG CONFIG

This should never be modified as it is generated with regards to all other config params.
""" 

def get_experiment_name():
    basename= model['net'] + '_' + optim['type'] + '_' + scheduler['type'] + '_' + model['activation']
    if model['quantize']:
        basename += '_quant'
    if model['self_attention']:
        basename += '_sa'
    if model['label_smoothing']:
        basename += '_ls'
    if model['shakedrop']:
        basename += '_shkdrp'
    if dataloader['winner_config']:
        basename += '_wc'
    elif dataloader['data_aug']:
        if dataloader['fast_aug']:
            basename += '_faa'
        else:
            basename += '_aa'
    if dataloader['use_cutout']:
        basename += '_cutout'
    if dataloader['resize']:
        basename += '_resized'
    if train['use_cutmix']:
        basename += '_cutmix'
    if train['use_pruning']:
        basename += '_pruned'
    if train['use_binary_connect']:
        basename += 'bc'
    if train['distillation']:
        basename += 'student_fromTeacher_'+teacher['net']
    return basename

log = {
    'tensorboard_path': './runs/'+dataset+'/'+get_experiment_name(),
    'checkpoints_path': './checkpoints/'+dataset+'/'+get_experiment_name()+'.pt'
}
    