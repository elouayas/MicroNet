import torch.nn as nn
import numpy
import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans
import random
import sys

class K_means():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):# and m.weight.shape[3]==3 :# or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.kmeans=[]
        self.idx=[]
        #self.clusters=numpy.zeros(len(b)).cuda()
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):# and m.weight.shape[3]==3:# or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def quantization(self,c,k):#,CR):
        self.save_params()
        #self.idx=numpy.zeros((self.num_of_params,c,k))
        c_tmp=c
        for index in range(self.num_of_params):
            print(index)
            li=[]        
            a=self.target_modules[index].clone().cpu()
            #print(a.shape[1])
            if(index==0 or index==2):
                c=16
            else:
                c=c_tmp
            a=a.view(a.shape[3]*a.shape[2]*a.shape[0],a.shape[1])
            #k=a.shape[0]//CR
            size=a.shape[1]//c
            for j in range(c):
                li2=[]             
                self.kmeans.append(KMeans(n_clusters=k, random_state=0,n_jobs=-1).fit(a[:,j*size:(j+1)*size].data))
                for kk in range(k):
                    li2.append([x for x in range(len(self.kmeans[-1].labels_)) if self.kmeans[-1].labels_[x]==kk])
                li.append(li2)
                for gh in range(a.shape[0]):
                    a[gh,j*size:(j+1)*size].data.copy_(torch.from_numpy(self.kmeans[-1].cluster_centers_[self.kmeans[-1].labels_[gh]]))
            a=a.view(self.target_modules[index].shape[0],self.target_modules[index].shape[1],self.target_modules[index].shape[2],self.target_modules[index].shape[3])
            self.target_modules[index].data.copy_(a.data)
                
            self.idx.append(li)#=self.idx.tolist()
    

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].data.copy_(self.target_modules[index].data)

            
    def restore(self,c,k):
        
        idx=-1
        c_tmp=c
        for index in range(self.num_of_params):
            a=self.target_modules[index]-self.saved_params[index]
            if(index==0 or index==2):
                c=16
            else:
                c=c_tmp
            a=a.view(a.shape[0]*a.shape[3]*a.shape[2],a.shape[1])
            #k=a.shape[0]//CR
            size=a.shape[1]//c
            sum_grad=torch.zeros(k,a.shape[1])#*a.shape[2]*a.shape[3]//c)
            
            sum_grad=sum_grad.cuda()
            for j in range(c):
                idx+=1
                for kk in range(k):  
                #for i in range(a.shape[0]):
                    sum_grad[kk,j*size:(j+1)*size].data.copy_(torch.sum(a[self.idx[index][j][kk],j*size:(j+1)*size].data,0))
                #sum_grad[self.kmeans[idx].labels_].data.copy_(sum_grad[self.kmeans[idx].labels_].data+a[:,j*size:(j+1)*size].data)
                
                #for i in range(a.shape[0]):
                a[:,j*size:(j+1)*size].data.copy_(sum_grad[self.kmeans[idx].labels_,j*size:(j+1)*size])
                #a.data.copy_(sum_grad[self.kmeans[idx].labels_])
            #for i in range(a.shape[0]):
            #    for j in range(a.shape[0]):
            #        if(self.kmeans[index].labels_[i]==self.kmeans[index].labels_[j]):
            #            count[i].data.copy_(count[i].data+a[j].data)
                        #print(i)
            
            #for i in range(a.shape[0]):
            #    a[i].data.copy_(count[i])#a[i].data*count[self.kmeans[index].labels_[gh]])
            
            a=a.view(self.target_modules[index].shape[0],
                   self.target_modules[index].shape[1],self.target_modules[index].shape[2],self.target_modules[index].shape[3])
            
            self.target_modules[index].data.copy_(a.data+self.saved_params[index])

    def clip(self):
        clip_scale=[]
        m=nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)
    def calcul_rapport(self,c,k,logk):
        a=0
        rapport1=0 #6400+432
        rapport2=0 #6400+432
        c_tmp=c
        for index in range(self.num_of_params):
            if(index==0 or index==2):
                c=16
            else:
                c=c_tmp
            r1=self.target_modules[index].shape[0]*self.target_modules[index].shape[1]*self.target_modules[index].shape[2]*self.target_modules[index].shape[3]
            r2=self.target_modules[index].shape[0]*self.target_modules[index].shape[2]*self.target_modules[index].shape[3]
            rapport1+=r1*32
            rapport2+=k*self.target_modules[index].shape[1]*32 + r2*logk*c
            #rapport+=a/(a*k/(c*32)+c*logk)
        #rapport=(rapport)/(self.num_of_params)
        print(rapport1/rapport2)
        return rapport1/rapport2
