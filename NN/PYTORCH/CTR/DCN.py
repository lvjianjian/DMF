#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 19-4-10, 13:21

@Description:

@Update Date: 19-4-10, 13:21
"""

from DMF.NN.PYTORCH.CTR import *

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

    def _deep(self, x):
        return deep(x,self.deeps,'relu')

    def _cross(self, x):
        return cross(x, self.crosses)

    def forward(self, x):  # (batch_size, x_dim)
        x1 = self._deep(x)
        x2 = self._cross(x)
        x = torch.cat([x1, x2], 1)
        return x



