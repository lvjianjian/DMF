#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-9-27, 22:52

@Description:

@Update Date: 18-9-27, 22:52
"""

from DMF.NN.PYTORCH.CTR import *

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

