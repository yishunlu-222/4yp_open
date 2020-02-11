# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from backbone_class_2_ssh import resnet_1D_34
from torchsummary import summary
import pdb
class VladPooling(nn.Module):
    '''
    NetVlad, GhostVlad implementation
    '''
    def __init__(self, mode, k_centers,input_shape, g_centers=0 , dim = 512, num_classes = 20 , **kwargs):
        super(VladPooling, self).__init__(**kwargs)
        self.dim = dim
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        self.centroids= nn.Parameter(torch.rand(k_centers+g_centers,dim))  # bz x W x H x (K+G) x D
        # print(k_centers)
        # print(g_centers)
        # print(k_centers+g_centers)
        self.fc = nn.Linear(dim , k_centers+g_centers)
        self.fc2 = nn.Linear(input_shape*k_centers, num_classes)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self,x):
        # feature : bz x 1 x 512 x T, cluster: bz  x 1 x clusters x T.
        feature, clusters_score = x
        # num_features = feature.shape[-2]
        # assert num_features == self.dim ,"feature dim not correct"

        # softmax of softassignment
        # feature = F.normalize(feature, p=2, dim=0)
        # print(feature.shape)
        # print(feature.permute(0,2,1).shape)
        soft_assign1 = self.fc(feature.permute(0, 2, 1))
        #(1,512,T)->(1,K+G,T) ->(1,T,K+G)
        soft_assign2 = F.softmax(soft_assign1,dim =1)  #(1,T,K+G)
        soft_assign = soft_assign2.unsqueeze(-1) #(1,T,K+G,1)

        # Now, need to compute the residual, clusters - x
        # self.centriod: clusters x D
        # need to have one more dimension to do sum(intra-normalization)
        feature_expand = feature.permute(0, 2, 1).unsqueeze(-2)
        # (1 x 512 x T) ->(1 x T x 512)->(1 x T x 1 x 512)
        feat_residual = feature_expand - self.centroids  #1 x T x (K+G) x D
        residual =  torch.mul(feat_residual,soft_assign) #1 x T x (K+G) x D
        vlad = torch.sum(residual,dim = -1)    #1 x T x (K+G)
        if self.mode == 'gvlad':
            vlad = vlad[:,:,:self.k_centers]    #B x T x (K) x D

        vlad_output = F.normalize(vlad, p=2, dim=1)
        vlad_output = vlad_output.reshape(vlad_output.shape[0], -1)  # B x T x (K x D）
        # pdb.set_trace()
        vlad_output = self.fc2(vlad_output)
        # pdb.set_trace()
        return vlad_output


class model(nn.Module):
    def __init__(self,mode,num_classes,input_shape,k_centers=8,g_centers=2,dim = 512,kernel_size= 100):
        super(model, self).__init__()
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.cnn = resnet_1D_34(kernel_size=self.kernel_size)
        self.vlad = VladPooling( mode = mode, k_centers = k_centers,num_classes = num_classes,
                                 g_centers = g_centers,dim = dim,input_shape =input_shape)
        self.convsoft = nn.Conv1d(in_channels= self.input_shape,out_channels=(k_centers + g_centers)* dim,
                                  bias=True,kernel_size = 1)

    # .weight = torch.autograd.Variable(torch.randn( k_centers + g_centers,dim,1)).to( device ) #

    def forward(self, x):
        x = self.cnn(x)
        # pdb.set_trace()
        x1 = x.permute(0, 2, 1)
        # cluster_score = .conv1d(x,self.weight)
        cluster_score = self.convsoft(x1)
        vlad_input  = [x, cluster_score]
        # pdb.set_trace()
        output = self.vlad(vlad_input)

        return output


if __name__ == '__main__':
    # a = resnet_1D_34()
    # print(a)
    # summary(a,input_size=(1,1,22000))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = resnet_1D_34(10)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # net = nn.DataParallel(net)
    else:
        print("Let's use", torch.cuda.device_count(), "GPU!")
    net.to(device)
    my_tensor = torch.randn(1, 1, 48000)
    my_tensor = my_tensor.to(device)
    y = net(my_tensor)
    torch.cuda.empty_cache()
    # input_shape: calculate the output shape from the cnn
    g_centres = 2
    print(y.shape
    )
    model = model(mode='gvlad', k_centers=8, g_centers=g_centres, dim=256, input_shape=y.shape[-1]
                  ,kernel_size= 10,num_classes = 200)
    model.to(device)
    # y = net(my_tensor)
    # # print(y)
    # print('y is {0}'.format(y.shape))
    # filter=torch.autograd.Variable(torch.randn(8+2,512,1)).to(device)
    # cluster_score = F.conv1d(y,filter)
    # print('cluster_score is {0}'.format(cluster_score.shape))
    # Vlad = VladPooling( mode = 'gvlad', k_centers=8, g_centers=2,dim = 512,input_shape=y.shape).to(device)
    # output = Vlad([y,cluster_score])   # [1, 84808] means there are 84808 features in a audio
    output =model(my_tensor)
    pdb.set_trace()
    print(output.shape)
    # summary(model, input_size=(1, 30000))

