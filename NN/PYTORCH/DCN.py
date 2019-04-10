#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 19-4-10, 13:21

@Description:

@Update Date: 19-4-10, 13:21
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class DCN(nn.Module):
    def __init__(self, x_dim, deep_n, cross_n, deep_n_hidden):
        super(DCN, self).__init__()
        self.deep_n = deep_n  # 2-5
        self.deep_n_hidden = deep_n_hidden  # 32-1024
        self.cross_n = cross_n  # 1-6
        self.x_dim = x_dim
        self.deeps = nn.ModuleList()
        self.deeps.append(nn.Linear(x_dim, deep_n_hidden))
        for _i in range(1,deep_n):
            self.deeps.append(nn.Linear(deep_n_hidden, deep_n_hidden))

        self.crosses = nn.ModuleList()
        for _i in range(0, cross_n):
            self.crosses.append(nn.Linear(x_dim, x_dim))

    def deep(self, x):
        for _i in range(self.deep_n):
            x = self.deeps[_i](x)
            x = F.relu(x)
        return x

    def _cross(self, x):
        return cross(x, self.crosses)

    def forward(self, x):  # (batch_size, x_dim)
        x1 = self.deep(x)
        x2 = self._cross(x)
        x = torch.cat([x1, x2], 1)
        return x

def cross(x, crosses):
    x_l_1 = x
    for _i in range(len(crosses)):
        xl = (x.unsqueeze(1) * x_l_1.unsqueeze(2)).sum(2)
        xl = crosses[_i](xl) + x_l_1  # 可以尝试加激活函数
        x_l_1 = xl
    return x_l_1

def cross_relu(x, crosses):
    x_l_1 = x
    for _i in range(len(crosses)):
        xl = (x.unsqueeze(1) * x_l_1.unsqueeze(2)).sum(2)
        xl = crosses[_i](xl) + x_l_1
        x_l_1 = F.relu(xl, inplace=True)
    return x_l_1