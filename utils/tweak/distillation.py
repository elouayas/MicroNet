import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch 
import numpy as np

import torchvision
import torchvision.transforms as transforms


def get_distances(representations):
    rview = representations.view(representations.size(0),-1)
    distances = torch.cdist(rview,rview,p=2)
    return distances



def representations_to_adj(representations, k=128, A_final=None,mult=None):
    rview = representations.view(representations.size(0),-1)
    rview =  torch.nn.functional.normalize(rview, p=2, dim=1)
    adj = torch.mm(rview,torch.t(rview))
    ind = np.diag_indices(adj.shape[0])
    adj[ind[0], ind[1]] = torch.zeros(adj.shape[0]).cuda()
    degree = torch.pow(adj.sum(dim=1),-0.5)
    degree_matrix = torch.diag(degree)
    adj = torch.matmul(degree_matrix,torch.matmul(adj,degree_matrix))
    if type(mult) == torch.Tensor:
        adj = adj*mult
    if k != 128:
      if type(A_final) == torch.Tensor:
        adj = adj*A_final
      else:
        y, ind = torch.sort(adj, 1)
        A = torch.zeros(*y.size()).cuda()
        k_biggest = ind[:,-k:].data
        for index1,value in enumerate(k_biggest):
            A_line = A[index1]
            A_line[value] = 1
        A_final = torch.min(torch.ones(*y.size()).cuda(),A+torch.t(A))
        adj = adj*A_final

    return adj,A_final



def to_one_hot(inp,num_classes):
    y_onehot = torch.cuda.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    
    return y_onehot



class BatchMeanCrossEntropyWithLogSoftmax(nn.Module):
    def forward(self, y_hat, y):
        return -(y_hat*y).sum(dim=1).mean(dim=0)

class BatchMeanKLDivWithLogSoftmax(nn.Module):
    def forward(self, p, log_q,  log_p):
        return (p*log_p - p*log_q).sum(dim=1).mean(dim=0)


class CrossEntropyWithLogSoftmax(nn.Module):
    def forward(self, y_hat, y):
        return -(y_hat*y).mean()
    
    
def do_gkd(pool3_only, layers, teacher_layers, k, power, intra_only, lambda_gkd, intra_class, inter_only, inter_class):
    loss_gkd = 0 
    zips = zip(layers,teacher_layers) if not pool3_only else zip([layers[-1]],[teacher_layers[-1]])
    mult = None
    if intra_only:
        mult=intra_class
    elif inter_only:
        mult=inter_class
    for student_layer,teacher_layer in zips:
        adj_teacher,A_final = representations_to_adj(teacher_layer,k,mult=mult)
        adj_student,A_final = representations_to_adj(student_layer,k,A_final,mult=mult)
        adj_teacher_p = adj_teacher
        adj_student_p = adj_student
        for _ in range(power-1):
            adj_teacher_p = torch.matmul(adj_teacher_p,adj_teacher)
            adj_student_p = torch.matmul(adj_student_p,adj_student)
        loss_gkd += lambda_gkd*F.mse_loss(adj_teacher_p, adj_student_p, reduction='none').sum()
    return loss_gkd