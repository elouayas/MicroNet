import torch
import numpy as np
from scipy.optimize import fsolve # use to determine asymptotic rate per epoch

import config as cfg


class Mask(object):

    def __init__(self, net, nb_epochs):
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        self.config, self.nb_epochs = cfg.config, nb_epochs
        self.net_size, self.net_length = self._init_length() # size (eg (64,64,3,3)) and length (eg 64*64*3*3) per layer
        self.compress_rate = {} # compress rate per layer
        self.mat = {}
        self.mask_index = []
        
    
    def _init_length(self):
        ''' Initializes the length of each layer. '''
        net_size = {}
        for index, item in enumerate(self.net.parameters()):
            net_size[index] = item.size()
        net_length = {}
        for layer_index in net_size:
            for dim_index in range(0, len(net_size[layer_index])):
                if dim_index == 0:
                    net_length[layer_index] = net_size[layer_index][0]
                else:
                    net_length[layer_index] *= net_size[layer_index][dim_index]
        return net_size, net_length

    
    def _asymptotic_rate(self, epoch):
        def equations(p, P_min, P_goal, D, N_epoch):
            a, b, k = p
            equation_1 = a + b - P_min
            equation_2 = a * np.exp(-k*N_epoch)   + b - P_goal
            equation_3 = a * np.exp(-k*D*N_epoch) + b - (3/4) * P_goal
            return (equation_1, equation_2, equation_3)
        min_rate, goal_rate, D, nb_epochs = self.config['min'], self.config['goal'], self.config['D'], self.nb_epochs
        a, b, k =  fsolve(equations, (1, 1, 0), (min_rate, goal_rate, D, nb_epochs))
        def pruning_rate(epoch):
            return a*np.exp(-k*epoch)+b
        return pruning_rate(epoch)
        
                    
    def _init_rate(self, epoch):
        ''' Initializes the compression rate of each layer. '''
        # those are made for resnet18
        # last_index includes last fully connected layer.
        layer_begin, layer_inter, layer_end =  0, 3, 57
        last_index = 60
        skip_list = [21, 36, 51]
        for index, item in enumerate(self.net.parameters()):
            self.compress_rate[index] = 1
        if self.config['asymptotic']:
            pruning_rate = self._asymptotic_rate(epoch)
        else:
            pruning_rate = self.config['pruning_rate']
        for key in range(layer_begin, layer_end + 1, layer_inter):
            self.compress_rate[key] = pruning_rate
        self.mask_index = [x for x in range(0, last_index, layer_inter)]
        # Skips downsample layer.
        for x in skip_list:
            self.compress_rate[x] = 1
            self.mask_index.remove(x)


    def _get_filter_codebook(self, weight_torch, compress_rate, length):
        ''' Gets filter codebook. '''
        codebook = np.ones(length)
        # filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
        filter_pruned_num = int(weight_torch.size()[0] * compress_rate)
        weight_vec = weight_torch.view(weight_torch.size()[0], -1)
        norm2 = torch.norm(weight_vec, 2, 1)
        norm2_np = norm2.cpu().numpy()
        filter_index = norm2_np.argsort()[:filter_pruned_num]
        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        for x in range(0, len(filter_index)):
            codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
        codebook = torch.FloatTensor(codebook)
        codebook = codebook.to(self.device)
        return codebook


    def init_mask(self, epoch):
        ''' Initializes the mask. '''
        self._init_rate(epoch)
        for index, item in enumerate(self.net.parameters()):
            if index in self.mask_index:
                self.mat[index] = self._get_filter_codebook(item.data, self.compress_rate[index], self.net_length[index])

        
    def do_mask(self):
        ''' Performs pruning. '''
        for index, item in enumerate(self.net.parameters()):
            if index in self.mask_index:
                a = item.data.view(self.net_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.net_size[index])


    def verbose(self):
        ''' Prints information about network weights. '''
        if self.config['verbose'] != 0:
            layer_begin, layer_inter, layer_end =  0, 3, 57 # resnet18 constants
            nonzero_count, zero_count = 0, 0
            for index, item in enumerate(self.net.parameters()):
                if index in [x for x in range(layer_begin, layer_end + 1, layer_inter)]:
                    a = item.data.view(self.net_length[index])
                    b = a.cpu().numpy()
                    nb_nonzero = np.count_nonzero(b)
                    nb_zero = len(b) - np.count_nonzero(b)
                    nonzero_count += nb_nonzero
                    zero_count += nb_zero
                    if self.config['verbose'] == 2:
                        print('Layer: %d: %d nonzero weight, %d zero weight' % (index, nb_nonzero, nb_zero))
            if self.config['verbose'] == 1:
                print('Pruning: %d nonzero weight, %d zero weight' % (nonzero_count, zero_count))