"""
Adapted from https://github.com/wps712/MicroNetChallenge/tree/cifar100
"""

from numpy.random import choice as random_choice
from utils.data.transform import apply_policy, zero_pad_and_crop, random_flip, cutout_numpy
from torchvision import transforms

""" Default winner policies """

default_policies = [
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
 [("ShearX", 0.1, 3)]
]


class AutoAugment(object):
  
  def __init__(self, policies, num = 2):
    self.good_policies = policies
    self.num = num
    
  def __call__(self, data):
    x = data
    for i in range(self.num):
      epoch_policy = self.good_policies[random_choice(len(self.good_policies))]
      x = apply_policy(epoch_policy, data)
    x = zero_pad_and_crop(x, 4)
    x = random_flip(x)
    x = cutout_numpy(x)
    return x
    