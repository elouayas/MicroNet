"""
Generic Configuration File For Training Pipeline on CIFAR10/100
"""
from brevitas.core.quant import QuantType

########################################## DATASET ############################################

dataset = 'cifar100' # 'cifar10' or 'cifar100'

############################################ MODEL ############################################

# net can be one of:
# resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202, efficientnetb0, wide_resnet_28_10,pyramidnet272
# Modify utils/init_net.py to add new net
# See models/ to check which net are availables

model_config = {
    'net': 'densenet_100_micronet',
    'mode': 'baseline', # can be 'baseline' or 'boosted'
    'activation': 'ReLU', # can be 'ReLU' or 'Mish'
    # baseline means SGD + ROP, 'boosted' means RangerLars + DelayedCosineAnnealingLR
    # if lalbe_smoothing is True
    # use utils.LabelSmoothingCrossEntropy instead of vanilla CrossEntropy.
    'label_smoothing': False,
    # if smoothing = x, final label value will be 1-x instead of 1 with vanilla CrossEntropy.
    'smoothing': 0.1,
    'reduction': 'mean', # loss reduction. Can be 'mean' or 'sum'.
    # See the reduce_loss method in utils.label_smoothing for more info
    'quantize': False,
    'weight_quant_type': QuantType.INT, # can be BINARY, TERNARY, INT, FP
    'weight_bit_width': 8
}

######################################### DATALOADER ##########################################

# The default data augmentation parameters can be modified in dataloaders.py
# via the aug variable in get_transforms()
# Cutout parameters should be modified only in the following dictionaries.

dataloader_config = {
    'rootdir': './data/',
    'download': True,
    'batch_size': 128,
    'nb_workers': 6,
    'data_aug': False,
    'use_cutout': False,
    'n_holes': 1,
    'length': 16,
    'resize': False,
    'use_fastaugm': False
}


#########################################  PRUNING  ###########################################

pruning_config = {
    'use_pruning': False,
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


########################################### TRAIN #############################################

EXPERIMENT_NAME_SUFFIX = '_forCifar'

train_config = {
    'nb_epochs' : 1000,
    'tensorboard_path': './runs/'+dataset+'/'+ model_config['net']+'_'+model_config['mode']+ \
    EXPERIMENT_NAME_SUFFIX+'/',
    'checkpoints_path': './checkpoints/'+dataset+'/',
    'use_early_stopping': False,
    'patience': 70, # early stopping patience
    'delta': 0.01, # value of loss decrease to cancel early stopping
    'use_binary_connect': False,
    'pruning': pruning_config,
    'verbose': True,
    'distillation':True
}


############################################ Teacher ############################################

teacher_config  = {
    'teacher_path':'model_best.pth.tar',
    'lambda_hkd':10,
    'lambda_gkd':0,
    'lambda_rkd':0,
    'pool3_only':False,
    'temp':4,
    'power':1,
    'k':128,
    'intra_only':False,
    'inter_only':False,
    'net': 'PyramidNet200',
}


### CHANGER TEACHER CONFIG, POUR UTILISER LA CLASSE MODEL DANS LE TRAINER
#'teacher_path':'model_best.pth.tar', pyram200
#'cifar100_pyramid272_top1_11.74.pth' pyram272
    