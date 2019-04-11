#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 19-4-11, 11:32

@Description:

@Update Date: 19-4-11, 11:32
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class Embedding3D(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'embedding_dim2', 'padding_idx', 'max_norm',
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

def by_activation(x, activation):
    if(activation == 'relu'):
        x = F.relu(x, inplace=True)
    elif(activation == 'sigmoid'):
        x = F.sigmoid(x)
    elif(activation == 'tanh'):
        x = F.tanh(x)
    elif(activation == 'identity'):
        pass
    else:
        raise Exception('Error activation: ', activation)
    return x

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

def deep(x, deeps, activation='identity'):
    '''
    :param x: (batch_size, x_dim)
    :param deeps: multi nn.Linear
    :param activation:
    :return: (b, last linear hidden)
    '''
    for _nn in range(deeps):
        x = _nn(x)
        x = by_activation(x,activation)
    return x

def cross(x, crosses, activation='identity'):
    '''
    :param x: (batch_size, x_dim)
    :param crosses: multi nn.Linear(x_dim, x_dim)
    :return: (b, x_dim)
    '''
    x_l_1 = x
    for _i in range(len(crosses)):
        xl = (x.unsqueeze(1) * x_l_1.unsqueeze(2)).sum(2)
        xl = crosses[_i](xl) + x_l_1
        x_l_1 = by_activation(xl, activation)
    return x_l_1

def cin(x, cins, activation = 'identity'):
    '''

    :param x: # (b, f, k)
    :param cins: multi paramaters
    :param activation:
    :return: (b, hidden, k), hidden is cin hidden
    '''
    x_l_1 = x
    for _item in cins:
        temp = x_l_1.unsqueeze(1) * x.unsqueeze(2)
        temp = temp.permute(0,3,2,1)
        temps = []
        for _w in _item:
            temps.append((temp * _w).permute(0,3,2,1).sum([1,2]).unsqueeze(1))
        x_l_1 = torch.cat(temps, 1)
        x_l_1 = by_activation(x_l_1, activation)
    return x_l_1

def cross_cnn(x, cnns, activation='identity'):
    '''
    replace cin for fast training
    :param x: (b, f, k)
    :param cnns: multi conv2d
    :param activation:
    :return: (b, hidden, k), hidden is cin hidden
    '''
    x_l_1 = x
    for conv in cnns:
        temp = x_l_1.unsqueeze(1) * x.unsqueeze(2) # (b,f,h,k)
        temp = conv(temp)  # (b,h,h,k)
        temp = temp.sum(2)
        x_l_1 = by_activation(temp, activation)
    return x_l_1  # (b, hidden, k)