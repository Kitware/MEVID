import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import model

class SpatialMemory(nn.Module):
    def __init__(self, feat_dim=2048, mem_size=100, margin=0.3):
        super(SpatialMemory, self).__init__()
        self.key = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.val = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.bn = nn.BatchNorm1d(feat_dim)
        nn.init.constant_(self.bn.weight, 0)
        nn.init.constant_(self.bn.bias, 0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = feat_dim
        self.margin = margin

    def forward(self, query, val):
        BS, self.feat_dim, H, W = query.shape
        query_rs = query.reshape(-1, self.feat_dim)

        similarity = torch.matmul(F.normalize(query_rs, dim=1),
                                  F.normalize(self.key.t(), dim=1))
        r_att = F.softmax(similarity, dim=1)
        read = torch.matmul(r_att, self.val)
        read_ = self.bn(read)

        out = val - read_.reshape(BS, self.feat_dim, H, W)
        out = self.avgpool(out).squeeze()
        return {'out':out, 'loss':self.loss(r_att, self.margin)}

    def loss(self, r_att, margin):
        topk = r_att.topk(r_att.shape[0], dim=0)[0]
        distance = topk[-1] - topk[0] + margin
        mem_trip = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return {'mem_trip':mem_trip}

class TemporalMemory(nn.Module):
    def __init__(self, feat_dim=2048, mem_size=100, margin=1, seq_len=6):
        super(TemporalMemory, self).__init__()
        self.key = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.val = nn.Parameter(torch.empty(mem_size, seq_len).uniform_().cuda())
        self.lstm = nn.LSTM(feat_dim, feat_dim, 1)
        self.margin = margin
        self.S = seq_len

    def forward(self, query, val):
        query = query.reshape(query.shape[0]//self.S, self.S, -1).permute(1, 0, 2)
        h0 = torch.zeros(1, query.shape[1], query.shape[2]).cuda()
        c0 = torch.zeros(1, query.shape[1], query.shape[2]).cuda()
        if self.training: self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(query, (h0, c0))
        query_lstm = output[-1] # [B, F]

        similarity = torch.matmul(F.normalize(query_lstm, dim=1),
                                  F.normalize(self.key.t(), dim=1))
        r_att = F.softmax(similarity, dim=1)
        read = F.softmax(torch.matmul(r_att, self.val), dim=1)

        val = val.reshape(val.shape[0]//self.S, self.S, -1)
        out = torch.bmm(read.unsqueeze(1), val).squeeze(1)
        return {'out':out, 'loss':self.loss(r_att, self.margin)}

    def loss(self, r_att, margin=1):
        topk = r_att.topk(r_att.shape[0], dim=0)[0]
        distance = topk[-1] - topk[0] + margin
        mem_trip = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return {'mem_trip':mem_trip}
