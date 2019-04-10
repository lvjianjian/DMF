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

class FM(nn.Module):
    def __init__(self, n, k, field_size):
        super(FM, self).__init__()
        self.n = n # feature size
        self.k = k # embedding size
        self.f = field_size
        self.linear = nn.Embedding(self.n, 1)
        self.v = nn.Embedding(self.n, self.k)
        self.b = nn.Parameter(torch.randn((1)).type(torch.FloatTensor), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def fm_layer(self, x):
        linear_part = self.linear(x)
        linear_part = torch.sum(linear_part,1) + self.b
        em = self.v(x)
        f_e_m_sum = torch.sum(em, 1)
        f_e_m_sum_square = torch.pow(f_e_m_sum, 2)
        f_e_m_square = torch.pow(em, 2)
        f_e_m_square_sum = torch.sum(f_e_m_square, 1)
        second_order = f_e_m_sum_square - f_e_m_square_sum
        second_order = torch.sum(second_order, 1, keepdim=True)
        logits = 0.5 * second_order + linear_part
        # logit = self.sigmoid(logit)
        del em,linear_part,f_e_m_sum,f_e_m_square,f_e_m_square_sum,second_order
        return logits  # out_size = (batch, 1)

    def forward(self, x):
        output = self.fm_layer(x)
        return output

    def predict(self, x):
        return self.sigmoid(self(x))

def fm_part1(x, use_cuda=True):
    '''
    :param x: (batch_size, field_size, k)
    :return: (batch_size, field_size * (field_size - 1) // 2)
    '''
    x = x.unsqueeze(2) * x.unsqueeze(1)
    x = x.sum(-1)
    mask = torch.ones_like(x)
    mask = torch.triu(mask[0,:,:],diagonal=1)
    if(use_cuda):
        x = torch.masked_select(x, mask.type(torch.cuda.ByteTensor)).view(x.size(0),-1)
    else:
        x = torch.masked_select(x, mask.type(torch.ByteTensor)).view(x.size(0),-1)
    return x

def fm_part2(x, use_cuda=True):
    '''
    :param x: (batch_size, field_size, k)
    :return: (batch_size, field_size * (field_size - 1) // 2 * k)
    '''
    x = x.unsqueeze(2) * x.unsqueeze(1)
    mask = torch.ones_like(x)
    mask = torch.triu(mask[0, :, :, 0], diagonal=1)
    mask = mask.repeat(x.size(0), 1, 1)
    mask = mask.unsqueeze(3)
    if(use_cuda):
        x = torch.masked_select(x, mask.type(torch.cuda.ByteTensor)).view(x.size(0), -1)
    else:
        x = torch.masked_select(x, mask.type(torch.ByteTensor)).view(x.size(0), -1)
    return x