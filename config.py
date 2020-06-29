""" Generic Configuration File For Training Pipeline on CIFAR10/100

This file will be used to set an instance of the Model class from model.py

Here are defined 7 Dataclasses:
    * Dataset
    * Dataloader
    * Model
    * Optimizer
    * Scheduler
    * Train
    * Distillation

A final "Meta Data Class" Config is defined, which contains the 7 dataclasses.
This meta dataclass will be used to instanciate a LightningModel.
"""

from dataclasses import dataclass, field
from utils.data.policies import AUGMENTATION_POLICIES

# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                      DATASET CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Dataset:
    """
    Args:

        name (str):  'cifar10' or 'cifar100'.
                      This will affect:
                        * the dataloaders in utils.data.dataloader.py
                        * the final layer of model (via the num_classes param
    """

    name: str = 'cifar100'




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                   DATALOADER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Dataloader:
    """
    Params to generate augmented dataloaders.
    Basicaly, it will randomly apply a specified number of policies.

    Args:

        rootdir (str):  path to the dataset

        download (bool): Download data if not found in rootdir.

        train_batch_size (int): Batch size for the training dataloader.

        val_batch_size (int): Batch size for the validation dataloader.

        nb_workers (int): number of threads to load data.
                        Param for the vanilla Pytorch Dataloader class.
                        Should be nb_gpus_available * 4

        policies (dict): Data-aug policy of MicroNet 2019 winners:
                        https://github.com/wps712/MicroNetChallenge/tree/cifar100
                        Refer to utils.data.policies to see the default winner policies.
                        Set cutout in AUGMENTATION_POLICIES as (0,0) when using cutmix.

        augnum (int): number of policies to use to augment data
    """

    rootdir: str = './data/'
    download: bool = False
    train_batch_size: int = 32
    val_batch_size: int = 32
    nb_workers: int = 4
    policies: list = field(default_factory = lambda: AUGMENTATION_POLICIES)
    augnum: int = 2




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                      MODEL CONFIG                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Model:
    """
    In all the code we call a model the set of:
        * a network
        * an optimizer
        * a scheduler
        * a criterion (a loss)
    This is consistant with the Pytorch Lightning philosophy.
    This is somewhat unusual and one must remind the definition of model and net here.

    Args:

        net (str):  defines the network to train.
                    Can be 'densenet100' or 'densenet172'.

        activation (str): the activation function to use. Can be one of:
                        * ReLU
                        * Swish
                        * Mish
                        Mish seems to often gives best results.
                        Be careful to adjust the learning rate accordingly.
                        One can use the learning rate finder option for Pytorch Lightning.

        self_attention (bool): boolean to control the insertion of a simple self attention layer
                            inside the basic blocks of the network.
                            See https://github.com/sdoria/SimpleSelfAttention
                            Be careful to adjust the learning rate accordingly.
                            One can use the learning rate finder option for Pytorch Lightning.

        attention_sym (bool): force or not the attention matrix to be symetric.
                            Having a symetric matrix help convergence.

        shakedrop (bool) : add shakedrop at the end of the forward method

        label_smoothing (bool): If True, use utils.model.layers.LabelSmoothingCrossEntropy
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
    """

    net: str = 'densenet172'
    activation: str = 'relu'
    self_attention: bool = False
    attention_sym: bool = False
    shakedrop: bool = False
    use_label_smoothing: bool = False
    smoothing: float = 0.1
    reduction: str = 'mean'




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    OPTIMIZER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Optimizer:
    """
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

    optim: str = 'SGD'
    params: dict = field(default_factory = lambda: {
        'SGD':    {'lr': 0.0001, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 5e-4},
        'RAlamb': {'lr': 1e-3,  'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay':0}
    })
    use_lookahead: bool = False
    lookahead: dict = field(default_factory = lambda: {'alpha': 0.5, 'k': 6})




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    SCHEDULER CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Scheduler:
    """
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

        use_delay: boolean controlling the use of Decay.
                If True, the scheduler defined will start being active after the given
                number of epochs.
                This is often usefull when using cosine or exponential annealing.

        delay: lookahead params. Must be a dict.
    """

    scheduler: str = 'Cosine'
    params: dict = field(default_factory = lambda: {
        'ROP'               : {'mode': 'min', 'factor': 0.2, 'patience': 20, 'verbose': True},
        'MultiStep'         : {'milestones': [120, 200], 'gamma': 0.1, 'last_epoch': -1},
        'Cosine'            : {'epochs': 300, 'eta_min': 0, 'last_epoch': -1},
        'WarmRestartsCosine': {'T_0': 150, 'T_mult': 1, 'eta_min': 0, 'last_epoch': -1}
    })
    use_warmup: bool = True
    warmup: dict = field(default_factory = lambda: {
        'multiplier': 1000, 'warmup_epochs': 5})
    use_delay: bool = False
    delay: dict = field(default_factory = lambda: {
        'delay_epochs': 150, 'after_scheduler': 'Cosine'})




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    TRAIN CONFIG                                     | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Train:
    """
    Args:

        nb_epochs (int): number of training epoch.
                        For each training epoch, one testing step is performed.

        use_cutmix (bool): controls the use of cutmix.
                        If True, Trainer.train() will call the Cutmix class
                        from utils.model.layers.cutmix with a probability p.
                        Set cutout in winner_policies as (0,0) when using Cutmix.

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

    nb_epochs: int = 300
    use_cutmix: bool = False
    cutmix_beta: float = 1.0
    cutmix_p: float = 0.5
    distillation: bool = False




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                 DISTILLATION CONFIG                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Distillation:
    """
    Specifies the kind of  distillation to use and its config.
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

    net: str = 'densenet172'
    teacher_path: str = ''
    lambda_hkd: int = 0
    lambda_gkd: int = 10
    lambda_rkd: int = 0
    temp: int = 4
    pool3_only: bool = False
    k: int = 128
    power: int = 1
    intra_only: bool = False
    inter_only: bool = False



# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                      META CONFIG                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class Config:
    """ Wrapp the 7 config Dataclasses into one Dataclass """

    dataset: Dataset()
    dataloader: Dataloader()
    model: Model()
    optimizer: Optimizer()
    scheduler: Scheduler()
    train: Train()
    distillation: Distillation()
