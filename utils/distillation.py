""" Class implementing 3 knowledge distillation methods """

import torch
import torch.nn.functional as F
import numpy as np

from utils.model.layers.criterion import (BatchMeanCrossEntropyWithLogSoftmax,
                                          BatchMeanKLDivWithLogSoftmax)



class Distillator():

    """ Adapted from https://arxiv.org/abs/1911.03080 """

    def __init__(self, dataset, teacher, config):
        self.dataset    = dataset
        self.teacher    = teacher.net
        self.lambda_hkd = config.lambda_hkd
        self.lambda_gkd = config.lambda_gkd
        self.lambda_rkd = config.lambda_rkd
        self.pool3_only = config.pool3_only
        self.temp       = config.temp
        self.power      = config.power
        self.k          = config.k
        self.intra_only = config.intra_only
        self.inter_only = config.inter_only

    def get_distances(self, representations):
        """ returns distance of network's representations """
        rview = representations.view(representations.size(0),-1)
        distances = torch.cdist(rview,rview,p=2)
        return distances

    def representations_to_adj(self, representations, adj_matrix=None, mult=None):
        """ from network's representations constructs a similarity matric """
        rview = representations.view(representations.size(0),-1)
        rview =  torch.nn.functional.normalize(rview, p=2, dim=1)
        adj = torch.mm(rview,torch.t(rview))
        ind = np.diag_indices(adj.shape[0])
        adj[ind[0], ind[1]] = torch.zeros(adj.shape[0]).cuda()
        degree = torch.pow(adj.sum(dim=1),-0.5)
        degree_matrix = torch.diag(degree)
        adj = torch.matmul(degree_matrix,torch.matmul(adj,degree_matrix))
        if isinstance(mult) == torch.Tensor:
            adj = adj*mult
        if self.k != 128:
            if isinstance(adj_matrix) == torch.Tensor:
                adj = adj*adj_matrix
            else:
                data, ind = torch.sort(adj, 1)
                canvas = torch.zeros(*data.size()).cuda()
                adj_matrix = torch.min(torch.ones(*data.size()).cuda(), canvas+torch.t(canvas))
                adj = adj*adj_matrix
        return adj,adj_matrix


    def to_one_hot(self, inpt, num_classes):
        """ classic one hot encoding of n  classes """
        y_onehot = torch.cuda.FloatTensor(inpt.size(0), num_classes)
        y_onehot.zero_()
        y_onehot.scatter_(1, inpt.unsqueeze(1), 1)
        return y_onehot

    def do_gkd(self, loss, layers, teacher_layers, intra_class, inter_class):
        """ perform a graph knowledge distillation (https://arxiv.org/abs/1911.03080) """
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
            adj_teacher, adj_matrix = self.representations_to_adj(teacher_layer, mult=mult)
            adj_student, adj_matrix = self.representations_to_adj(student_layer, adj_matrix,
                                                                  mult=mult)
            adj_teacher_p = adj_teacher
            adj_student_p = adj_student
            for _ in range(self.power-1):
                adj_teacher_p = torch.matmul(adj_teacher_p,adj_teacher)
                adj_student_p = torch.matmul(adj_student_p,adj_student)
            loss_gkd += self.lambda_gkd*F.mse_loss(adj_teacher_p, adj_student_p,
                                                   reduction='none').sum()
        loss += loss_gkd if self.pool3_only else loss_gkd/3
        return loss


    def do_hkd(self, loss, outputs, teacher_outputs):
        """ perform a classic knowledge distillation """
        proba = F.softmax(teacher_outputs/self.temp,dim=-1)
        log_q = F.log_softmax(outputs/self.temp,dim=-1)
        log_p = F.log_softmax(teacher_outputs/self.temp,dim=-1)
        hkd_loss = BatchMeanKLDivWithLogSoftmax()(p=proba,log_q=log_q,log_p=log_p)
        loss += self.lambda_hkd*hkd_loss
        return loss

    def do_rkd(self, loss, layers, teacher_layers):
        """ perform a relational knowledge distillation """
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
        return loss


    def run(self, inputs, outputs, labels, layers):
        """ takes the output of a forward pass in the student,
            and perform the knowledge distillation """
        num_classes = 10 if self.dataset == 'cifar10' else 100
        one_hot_labels = self.to_one_hot(labels, num_classes)
        intra_class = torch.matmul(one_hot_labels, one_hot_labels.T)
        inter_class = 1 - intra_class
        labels = one_hot_labels.argmax(dim=1)
        criterion = BatchMeanCrossEntropyWithLogSoftmax()
        loss = criterion(F.log_softmax(outputs,dim=-1),one_hot_labels)
        with torch.no_grad():
            teacher_output, teacher_layers = self.teacher(inputs)
            if self.lambda_hkd > 0:
                loss = self.do_hkd(loss, outputs, teacher_output)
            if self.lambda_rkd > 0:
                loss = self.do_rkd(loss, layers, teacher_layers)
            elif self.lambda_gkd > 0:
                loss = self.do_gkd(loss, layers, teacher_layers, intra_class, inter_class)
        return loss
