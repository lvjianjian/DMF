#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 19-4-3, 19:59

@Description:

@Update Date: 19-4-3, 19:59
"""
import torch.nn as nn
import torch

class Embedding3D(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim','embedding_dim2', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse', '_weight']

    def __init__(self, n, f, k):
        super(Embedding3D, self).__init__()
        self.n = n
        self.f = f
        self.k = k
        self.weight = nn.Parameter(torch.Tensor(n, f, k))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input):
        input = input.view(-1)
        input = torch.index_select(self.weight, 0, input).view(-1, self.f, self.f,self.k)
        return input

    def extra_repr(self):
        s = '{n}, {f}, {k}'
        return s.format(**self.__dict__)


class FFM1(nn.Module):# for each field just one value
    def __init__(self, n, k, field_size):
        super(FFM1, self).__init__()
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


def ffm_part(em, use_cuda = True):
    '''
    :param self:
    :param em: (batch_size, field_size, field_size ,k)
    :return:(field_size * (field_size-1) //2 * k)
    '''
    em2 = em.permute(0, 2, 1, 3)
    _temp = torch.ones_like(em)
    _temp = torch.triu(_temp[0, :, :, 0], diagonal=1)
    _temp = _temp.repeat(em.size(0), 1, 1)
    _temp = _temp.reshape(_temp.size(0), _temp.size(1), _temp.size(2), 1)
    em = em * em2 * _temp
    if(use_cuda):
        em2 = torch.masked_select(em, _temp.type(torch.cuda.ByteTensor)).view(em.size(0), -1)
    else:
        em2 = torch.masked_select(em, _temp.type(torch.ByteTensor)).view(em.size(0), -1)
    return em2