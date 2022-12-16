from __future__ import absolute_import

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class TKS(nn.Module):
    def __init__(self, in_channel,  **kwargs):
        super(TKS, self).__init__()
        reduction = 16
        squeeze_size = 8
        print('TSK, reduction: {}, squeeze_size: {}'.format(reduction, squeeze_size))
        d = in_channel//reduction
        k1 = squeeze_size
        k2 = squeeze_size // 2
        self.k1, self.k2 = k1, k2

        self.pool1 = nn.AvgPool3d(kernel_size=(1, k1, k1), stride=(1, k1, k1))
        self.pool2 = nn.AvgPool3d(kernel_size=(1, k2, k2), stride=(1, k2, k2))
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.conv1 = nn.Conv3d(in_channel, in_channel, kernel_size=(3, 1, 1), 
                padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(in_channel, in_channel, kernel_size=(3, 1, 1), 
                padding=(2, 0, 0), dilation=(2, 1, 1), bias=False)

        self.fc1 = nn.Sequential(nn.Conv3d(in_channel, d, 1, bias=False),
                                 nn.BatchNorm3d(d),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Conv3d(d, in_channel*2, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.W = nn.BatchNorm3d(in_channel)
        nn.init.constant_(self.W.weight.data, 0.0)
        nn.init.constant_(self.W.bias.data, 0.0)

    def forward(self, x1, x2):
        """
        x1: [bs, c, 2, 32, 16]
        x2: [bs, c, 4, 16, 8]
        """
        b, c, t1, h1, w1 = x1.size()
        b, c, t2, h2, w2 = x2.size()
        x1_in, x2_in = x1, x2

        x1 = self.pool1(x1) #[b, c, 2, 8, 4]
        x2 = self.pool2(x2) #[b, c, 4, 8, 4]

        x = torch.cat((x1[:,:,:1], x2[:,:,:3], x1[:,:,1:], x2[:,:,3:]), 2) 
        y1 = self.conv1(x) 
        y2 = self.conv2(x) 

        u = self.global_pool(y1 + y2) #[b, c, 1, 1, 1]
        s = self.fc1(u)
        a_b = self.fc2(s)
        a_b = a_b.view(b, 2, c, 1, 1, 1)
        a_b = self.softmax(a_b) 

        y = torch.stack((y1, y2), 1) 
        z = (y * a_b).sum(1)

        z1 = torch.cat((z[:,:,0:1], z[:,:,4:5]), 2) #[b, c, 2, h, w] 
        z2 = torch.cat((z[:,:,1:4], z[:,:,5:]), 2) #[b, c, K, h, w]

        z1 = z1.view(b, c, t1, z1.size(-2), 1, z1.size(-1), 1)
        z1 = z1.repeat(1, 1, 1, 1, self.k1, 1, self.k1)
        z1 = z1.view(b, c, t1, h1, w1)
        z1 = self.W(z1) + x1_in

        z2 = z2.view(b, c, t2, z2.size(-2), 1, z2.size(-1), 1)
        z2 = z2.repeat(1, 1, 1, 1, self.k2, 1, self.k2)
        z2 = z2.view(b, c, t2, h2, w2)
        z2 = self.W(z2) + x2_in
        return z1, z2