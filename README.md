# MicroNet
Pytorch implementation for the [MicroNet Challenge 2020](https://micronet-challenge.github.io/)

Hub for optimization techiques.

## Techniques implemented

1. Augmentation
    1. AutoAugment
    2. FastAugmentations
    3. Cutout
    4. CutMix

2. Scheduling
    1. RangerLars: RAdam + LARS + LookAHead
    2. Delayed LR Scheduler
    3. CosineAnnealingLR

3. Regularization
    1. Label Smoothing
    2. ShakeDrop
    3. Simple Self Attention

4. Distillation
    1. HKD
    2. RKD
    3. GKD

## Models availables

* resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
* densenet100, densenet121, densenet161, densenet169, densenet172, densenet201, 
* efficientnet-b0, ..., efficientnet_b7
* wide_resnet_28_10,
* pyramidnet200, pyramidnet272


## Organization

This code is based on two classes:
* Model
* Trainer



## Utilization

This github implements many optimization techniques and aims at being as modular as possible in the sens that one should be able to select which features to combine and launch a train flowlessy.

All hyperparameters can be configured in config.py
Y

### Config available

1. Dataset: cifar10 or cifar100
2. 


