#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 00:33:18 2018
based on DPN (Dual Path Network) implementation, https://github.com/oyam/pytorch-DPNs/blob/master/dpn.py

modified for audio genre task by Sungkyun Chang

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['DPN', 'dpn53', 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'dpns']


def dpn53(num_classes=16, center_dim=2):
    return DPN(num_init_features=64, k_R=96, G=32, k_sec=(3,4,7,3), inc_sec=(16,32,24,128), num_classes=num_classes,center_dim=center_dim)

def dpn92(num_classes=16, center_dim=2):
    return DPN(num_init_features=64, k_R=96, G=32, k_sec=(3,4,20,3), inc_sec=(16,32,24,128), num_classes=num_classes,center_dim=center_dim)


def dpn98(num_classes=16, center_dim=2):
    return DPN(num_init_features=96, k_R=160, G=40, k_sec=(3,6,20,3), inc_sec=(16,32,32,128), num_classes=num_classes,center_dim=center_dim)


def dpn131(num_classes=16, center_dim=2):
    return DPN(num_init_features=128, k_R=160, G=40, k_sec=(4,8,28,3), inc_sec=(16,32,32,128), num_classes=num_classes,center_dim=center_dim)


def dpn107(num_classes=16, center_dim=2):
    return DPN(num_init_features=128, k_R=200, G=50, k_sec=(4,8,20,3), inc_sec=(20,64,64,128), num_classes=num_classes,center_dim=center_dim)


dpns = {
    'dpn53': dpn53,
    'dpn92': dpn92,
    'dpn98': dpn98,
    'dpn107': dpn107,
    'dpn131': dpn131,
}


class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G, _type='normal'):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c

        if _type is 'proj':
            key_stride = 1
            self.has_proj = True
        if _type is 'down':
            key_stride = 2
            self.has_proj = True
        if _type is 'normal':
            key_stride = 1
            self.has_proj = False

        if self.has_proj:
            self.c1x1_w = self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_c+2*inc, kernel_size=1, stride=key_stride)

        self.layers = nn.Sequential(OrderedDict([
            ('c1x1_a', self.BN_ReLU_Conv(in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)),
            ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=key_stride, padding=1, groups=G)),
            ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+inc, kernel_size=1, stride=1)),
        ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        return nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)),
        ]))

    def forward(self, x):
        data_in = torch.cat(x, dim=1) if isinstance(x, list) else x
        if self.has_proj:
            data_o = self.c1x1_w(data_in)
            data_o1 = data_o[:,:self.num_1x1_c,:,:]
            data_o2 = data_o[:,self.num_1x1_c:,:,:]
        else:
            data_o1 = x[0]
            data_o2 = x[1]

        out = self.layers(data_in)

        summ = data_o1 + out[:,:self.num_1x1_c,:,:]
        dense = torch.cat([data_o2, out[:,self.num_1x1_c:,:,:]], dim=1)
        return [summ, dense]


class DPN(nn.Module):

    def __init__(self, num_init_features=64, k_R=96, G=32,
                 k_sec=(3, 4, 20, 3), inc_sec=(16,32,24,128), num_classes=16 ,center_dim=2):

        super(DPN, self).__init__()
        blocks = OrderedDict()

        # conv1: modified num of input channel 3->1
        blocks['conv1'] = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # conv2
        bw = 256
        inc = inc_sec[0]
        R = int((k_R*bw)/256)
        blocks['conv2_1'] = DualPathBlock(num_init_features, R, R, bw, inc, G, 'proj')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0]+1):
            blocks['conv2_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv3
        bw = 512
        inc = inc_sec[1]
        R = int((k_R*bw)/256)
        blocks['conv3_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1]+1):
            blocks['conv3_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv4
        bw = 1024
        inc = inc_sec[2]
        R = int((k_R*bw)/256)
        blocks['conv4_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2]+1):
            blocks['conv4_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        # conv5
        bw = 2048
        inc = inc_sec[3]
        R = int((k_R*bw)/256)
        blocks['conv5_1'] = DualPathBlock(in_chs, R, R, bw, inc, G, 'down')
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3]+1):
            blocks['conv5_{}'.format(i)] = DualPathBlock(in_chs, R, R, bw, inc, G, 'normal')
            in_chs += inc

        self.features = nn.Sequential(blocks)
        
        self.center_dim = center_dim
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(in_chs, self.center_dim) # input 2688, out 2
        self.classifier = nn.Linear(self.center_dim, num_classes) #


    def forward(self, x):
        out = self.features(x) #  {[2048,6,3] , [640, 6, 3]}
        #print(q.size())
        features = torch.cat(out, dim=1) # 2688,6,3
        #print(features.size())
        #features = torch.cat(self.features(x), dim=1) # modified avg_pool kernelsize 7x7 --> 6x3
        feat2d = F.avg_pool2d(features, kernel_size=(6,3)).view(features.size(0), -1) #batch_sz X 2688  # kernel_size was 7 in original dpn
        #p = F.avg_pool2d(features, kernel_size=(3,3), stride=3)
        #p = p.view(p.size(0), p.size(1), p.size(2)) # batch_sz X ch X feat_dim
        feat2d = self.preluip1(self.ip1(feat2d))
        out = self.classifier(feat2d)
        return F.log_softmax(out), feat2d