"""
Adapted from https://github.com/wps712/MicroNetChallenge/tree/cifar100
"""

from numpy.random import choice as random_choice
from utils.data.transform import apply_policy, zero_pad_and_crop, random_flip, cutout_numpy

class AutoAugment(object):
    """ Uses policies from utils.data.policies to augment data """

    def __init__(self, policies, num = 2):
        self.good_policies = policies
        self.num = num

    def __call__(self, data):
        for _ in range(self.num):
            epoch_policy = self.good_policies[random_choice(len(self.good_policies))]
            data = apply_policy(epoch_policy, data)
        data = zero_pad_and_crop(data, 4)
        data = random_flip(data)
        data = cutout_numpy(data)
        return data
