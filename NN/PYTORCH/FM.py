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


class FM(nn.Module):
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
        logitFM = (logitFM1 - logitFM2) * 0.5
        logit = (logitL + logitFM).squeeze(dim=-1).squeeze(dim=-1)
        # logit += self.B
        # logit = self.sigmoid(logit)
        return logit
