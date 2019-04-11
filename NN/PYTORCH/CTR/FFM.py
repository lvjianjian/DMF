#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 19-4-3, 19:59

@Description:

@Update Date: 19-4-3, 19:59
"""
from DMF.NN.PYTORCH.CTR import *

class FFM(nn.Module):# for each field just one value
    def __init__(self, n, k, field_size):
        super(FFM, self).__init__()
        self.n = n # feature size
        self.k = k # embedding size
        self.f = field_size
        self.linear = nn.Embedding(self.n, 1)
        self.v = Embedding3D(self.n, self.f, self.k)
        self.b = nn.Parameter(torch.randn((1)).type(torch.FloatTensor), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def ffm_layer(self, x):
        linear_part = self.linear(x)
        linear_part = torch.sum(linear_part, 1) + self.b
        em = self.v(x)
        em2 = em.permute(0,2,1,3)
        _temp = torch.ones_like(em)
        _temp = torch.triu(_temp[0,:,:,0],diagonal=1)
        _temp = _temp.repeat(em.size(0),1,1)
        _temp = _temp.reshape(_temp.size(0),_temp.size(1),_temp.size(2),1)
        em = em * em2 * _temp
        second_part = torch.sum(em,(1,2,3)).view(-1,1)
        logits = second_part + linear_part
        del linear_part,em,em2,_temp,second_part
        return logits  # out_size = (batch, 1)

    def forward(self, x):
        output = self.ffm_layer(x)
        return output


