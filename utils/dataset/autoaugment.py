"""
Adapted from https://github.com/wps712/MicroNetChallenge/tree/cifar100
"""

from numpy.random import choice as random_choice
from utils.dataset.transform import apply_policy, zero_pad_and_crop, random_flip, cutout_numpy
from torchvision import transforms


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
    