#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-9-27, 22:52

@Description:

@Update Date: 18-9-27, 22:52
"""

import torch.nn as nn
import torch
from torch.autograd import Variable


class FM(nn.Module):# have bug
    def __init__(self, args):
        self.model_name = "Factorization Machine"
        super(FM, self).__init__()
        self.nb_features = args['nb_features']
        self.dim_embed = args['dim_embed']
        self.embeddingL = nn.Embedding(self.nb_features, 1, padding_idx=None, max_norm=None, norm_type=2)
        self.embeddingQ = nn.Embedding(self.nb_features, self.dim_embed, padding_idx=None, max_norm=None, norm_type=2)
        # self.B = Variable(torch.randn((1)).type(torch.FloatTensor), requires_grad=True)
        # if (args['use_cuda']):
        #     self.B = self.B.cuda()
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        eL = self.embeddingL(x)
        logitL = eL.sum(dim=1, keepdim=True)
        eQ = self.embeddingQ(x)
        logitFM1 = eQ.mul(eQ).sum(1, keepdim=True).sum(2, keepdim=True)
        z = eQ.sum(dim=1, keepdim=True)
        z2 = z.mul(z)
        logitFM2 = z2.sum(dim=2, keepdim=True)
        logitFM = (logitFM2 - logitFM1) * 0.5
        logit = (logitL + logitFM).squeeze(dim=-1).squeeze(dim=-1)
        # logit += self.B
        # logit = self.sigmoid(logit)
        return logit


class FM_model(nn.Module):
    def __init__(self, n, k):
        super(FM_model, self).__init__()
        self.n = n # len(items) + len(users)
        self.k = k
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))

    def fm_layer(self, x):
        # x 属于 R^{batch*n}
        linear_part = self.linear(x)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k)
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        # 这里torch求和一定要用sum
        return output  # out_size = (batch, 1)

    def forward(self, x):
        output = self.fm_layer(x)
        return output