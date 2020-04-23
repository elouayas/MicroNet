import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch 
import numpy as np

import torchvision
import torchvision.transforms as transforms

from utils.layers.criterion import *



class Distillator():
    
    def __init__(self, dataset, teacher, teacher_config):
        self.dataset    = dataset
        self.teacher    = teacher.net
        self.lambda_hkd = teacher_config['lambda_hkd']
        self.lambda_gkd = teacher_config['lambda_gkd']
        self.lambda_rkd = teacher_config['lambda_rkd']
        self.pool3_only = teacher_config['pool3_only']
        self.temp       = teacher_config['temp']
        self.power      = teacher_config['power']
        self.k          = teacher_config['k']
        self.intra_only = teacher_config['intra_only']
        self.inter_only = teacher_config['inter_only']

    def get_distances(self, representations):
        rview = representations.view(representations.size(0),-1)
        distances = torch.cdist(rview,rview,p=2)
        return distances

    def representations_to_adj(self, representations, A_final=None, mult=None):
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
        if self.k != 128:
            if type(A_final) == torch.Tensor:
                adj = adj*A_final
            else:
                y, ind = torch.sort(adj, 1)
                A = torch.zeros(*y.size()).cuda()
                k_biggest = ind[:,-self.k:].data
                for index1,value in enumerate(k_biggest):
                    A_line = A[index1]
                    A_line[value] = 1
                A_final = torch.min(torch.ones(*y.size()).cuda(),A+torch.t(A))
                adj = adj*A_final
        return adj,A_final


    def to_one_hot(self, inpt, num_classes):
        y_onehot = torch.cuda.FloatTensor(inpt.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inpt.unsqueeze(1), 1)
        return y_onehot
    
    def do_gkd(self, loss, layers, teacher_layers, intra_class, inter_class):
        loss_gkd = 0
        if not self.pool3_only:
            zips = zip(layers,teacher_layers)
        else:
            zips = zip([layers[-1]],[teacher_layers[-1]]) 
        mult = None
        if self.intra_only:
            mult = intra_class
        elif self.inter_only:
            mult = inter_class
        for student_layer,teacher_layer in zips:
            adj_teacher,A_final = self.representations_to_adj(teacher_layer,
                                                              self.k,
                                                              mult=mult)
            adj_student,A_final = self.representations_to_adj(student_layer,
                                                              self.k,
                                                              A_final,
                                                              mult=mult)
            adj_teacher_p = adj_teacher
            adj_student_p = adj_student
            for _ in range(self.power-1):
                adj_teacher_p = torch.matmul(adj_teacher_p,adj_teacher)
                adj_student_p = torch.matmul(adj_student_p,adj_student)
            loss_gkd += self.lambda_gkd*F.mse_loss(adj_teacher_p,
                                                   adj_student_p,
                                                   reduction='none').sum()
        loss += loss_gkd if self.pool3_only else loss_gkd/3
        return loss


    def do_hkd(self, loss, outputs, teacher_outputs):
        p = F.softmax(teacher_outputs/self.temp,dim=-1)
        log_q = F.log_softmax(outputs/self.temp,dim=-1)
        log_p = F.log_softmax(teacher_outputs/self.temp,dim=-1)
        hkd_loss = BatchMeanKLDivWithLogSoftmax()(p=p,log_q=log_q,log_p=log_p)
        loss += self.lambda_hkd*hkd_loss
        return loss

    def do_rkd(self, loss, layers, teacher_layers):
        loss_rkd = 0
        if not self.pool3_only:
            zips = zip(layers,teacher_layers)
        else:
            zips = zip([layers[-1]],[teacher_layers[-1]])
        for student_layer,teacher_layer in zips:
            distances_teacher = self.get_distances(teacher_layer)
            distances_teacher = distances_teacher[distances_teacher>0]
            mean_teacher = distances_teacher.mean()
            distances_teacher = distances_teacher/mean_teacher
            distances_student = self.get_distances(student_layer)
            distances_student = distances_student[distances_student>0]
            mean_student = distances_student.mean()
            distances_student = distances_student/mean_student
            loss_rkd += self.lambda_rkd*F.smooth_l1_loss(distances_student,
                                                         distances_teacher,
                                                         reduction='none').mean()
        loss += loss_rkd if self.pool3_only else loss_rkd/3


    def run(self, inputs, outputs, labels):
        num_classes = 10 if self.dataset == 'cifar10' else 100 
        one_hot_labels = self.to_one_hot(labels, num_classes) 
        intra_class = torch.matmul(one_hot_labels, one_hot_labels.T)
        inter_class = 1 - intra_class
        labels = one_hot_labels.argmax(dim=1)
        criterion = BatchMeanCrossEntropyWithLogSoftmax()
        loss = criterion(F.log_softmax(outputs,dim=-1),one_hot_labels)
        with torch.no_grad():
            #TODO: adapter model pour qu'elle sorte output, layers
            # teacher_output, teacher_layers = teacher(inputs)"""
            self.teacher_output = self.teacher(inputs) # a supprimer quand la ligne au dessus sera fonctionel
            if self.lambda_hkd > 0:
                loss = self.do_hkd(loss, outputs, teacher_output)
            #if self.lambda_rkd > 0:
            #   loss = self.do_rkd(loss, layers, teacher_layers)
            #elif self.lambda_gkd > 0:
            #   loss = self.do_gkd(loss, layers, teacher_layers, intra_class, inter_class)
        return loss


