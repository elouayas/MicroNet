import random
import sys

import torch.nn as nn
import numpy
import torch

from torch.autograd import Variable
from sklearn.cluster import KMeans

from tqdm import tqdm


class K_means():
    def __init__(self, net):

        self.num_of_params  = self.count_layer_instances(net, nn.Conv2d) - 1
        self.saved_params   = []
        self.target_params  = []
        self.target_modules = []
        self.kmeans         = []
        self.idx            = []
        self.clone_layers(nn.Conv2d)

    def count_layer_instances(self, net, layer): 
        count_targets = 0
        for m in net.modules():
            if isinstance(m, layer): # and m.weight.shape[3]==3 :# or isinstance(m, nn.Linear):
                count_targets += 1
        return count_targets

    def clone_params_and_modules(self, net, layers):
        index = -1
        for m in net.modules():
            if isinstance(m, layers):# and m.weight.shape[3]==3:# or isinstance(m, nn.Linear):
                index = index + 1
                if index in range(1,self.num_of_params+1):
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def get_cluster_elements_indices(self, kmean, label):
        return [x for x in range(len(kmean.labels_)) if kmean.labels[x] == label]

    def get_cluster(self, kmean, idx):
        return kmean.cluster_centers_[kmean.labels_[idx]]

    def quantization(self, compression_factor, n_clusters):
        self.save_params()
        for i in range(self.num_of_params):
            li = []        
            target_module  = self.target_modules[i].clone().cpu()
            if i in [0,2]:
                compression_factor = 16
            out_channels, in_channels, kernel_h, kernel_w = target_module.shape
            target_module = target_module.view(out_channels*kernel_h*kernel_w,in_channels)
            size = in_channels // compression_factor
            for j in range(compression_factor):
                li2=[]             
                self.kmeans.append(KMeans(n_clusters=n_clusters,
                                          random_state=0,
                                          n_jobs=-1).fit(target_module[:,j*size:(j+1)*size].data))
                for label in range(n_clusters): # label are int in [0, n_clusters-1]
                    li2.append(self.get_cluster_elements_indices(self.kmeans[-1], label))
                li.append(li2)
                for channel_idx in range(out_channels):
                    target_module[channel_idx,j*size:(j+1)*size].data \
                    .copy_(torch.from_numpy(self.get_cluster(self.kmeans[-1], channel_idx)))
            target_module = target_module.view(out_channels, in_channels, kernel_h, kernel_w)
            self.target_modules[i].data.copy_(target_module.data) 
            self.idx.append(li)#=self.idx.tolist()
    
    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].data.copy_(self.target_modules[index].data)
  
    def restore(self, compression_factor, n_clusters):
        idx=-1
        for i in range(self.num_of_params):
            module = self.target_modules[i] - self.saved_params[i]
            if i in [0,2]:
                compression_factor = 16
            out_channels, in_channels, kernel_h, kernel_w = module.shape
            module = module.view(out_channels*kernel_h*kernel_w,in_channels)
            size = in_channels // compression_factor
            sum_grad = torch.zeros(n_clusters, in_channels)#*a.shape[2]*a.shape[3]//c)
            sum_grad = sum_grad.cuda()
            for j in range(compression_factor):
                idx+=1
                for k in range(n_clusters):  
                    sum_grad[k,j*size:(j+1)*size].data \
                    .copy_(torch.sum(module[self.idx[i][j][k],j*size:(j+1)*size].data,0))
                module[:,j*size:(j+1)*size].data. \
                copy_(sum_grad[self.kmeans[idx].labels_,j*size:(j+1)*size])
            module = module.view(out_channels, in_channels, kernel_h, kernel_w)
            self.target_modules[i].data.copy_(module.data+self.saved_params[i])

    def clip(self):
        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)
    
    def calcul_rapport(self, c, k, logk):
        a=0
        rapport1 = 0 #6400+432
        rapport2 = 0 #6400+432
        for index in range(self.num_of_params):
            if(index==0 or index==2):
                c = 16
            out_channels, in_channels, kernel_h, kernel_w = self.target_modules[index].shape
            r1 = out_channels * in_channels * kernel_w * kernel_w
            r2 = out_channels * kernel_h * kernel_w
            rapport1 += r1*32
            rapport2 += k * in_channels * 32 + r2*logk*c
        print(rapport1/rapport2)
        return rapport1/rapport2
