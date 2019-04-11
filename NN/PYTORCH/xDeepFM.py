#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 19-4-10, 14:33

@Description:

@Update Date: 19-4-10, 14:33
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class xDeepFM(nn.Module):
    def __init__(self, f, k, deep_n, deep_hidden, cin_n, cin_hidden):
        super(xDeepFM, self).__init__()
        self.deep_n = deep_n
        self.deep_hidden = deep_hidden
        self.f = f
        self.k = k
        self.cin_n = cin_n
        self.cin_hidden = cin_hidden
        self.deeps = nn.ModuleList()
        self.deeps.append(nn.Linear(f*k, deep_hidden))
        for _i in range(1, deep_n):
            self.deeps.append(nn.Linear(deep_hidden, deep_hidden))

        self.cins = nn.ModuleList()
        temp = nn.ParameterList()
        for _i in range(self.cin_hidden):
            temp.append(nn.Parameter(torch.Tensor(f, f)))
        self.cins.append(temp)
        for _i in range(1, self.cin_n):
            temp = nn.ParameterList()
            for _j in range(self.cin_hidden):
                temp.append(nn.Parameter(torch.Tensor(cin_hidden, f)))
            self.cins.append(temp)
        self.reset_parameters()

    def reset_parameters(self):
        for _item in self.cins:
            for _weight in _item:
                nn.init.normal_(_weight)

    def _cin(self, x):  # (b, f, k)
        x_l_1 = x
        for _item in self.cins:
            temp = x_l_1.unsqueeze(1) * x.unsqueeze(2)
            temp = temp.permute(0,3,2,1)
            temps = []
            for _w in _item:
                temps.append((temp * _w).permute(0,3,2,1).sum([1,2]).unsqueeze(1))
            x_l_1 = torch.cat(temps, 1)
        return x_l_1  # (b, hidden, k)

    def deep(self, x):
        x = x.view(x.size(0), -1)
        for m in self.deeps:
            x = m(x)
            x = F.relu(x)
        return x

    def forward(self, x):
        output1 = self._cin(x).view(x.size(0), -1)
        output2 = self.deep(x)
        return torch.cat([output1, output2], 1)

class xDeepCNN(nn.Module):
    def __init__(self, f, k, deep_n, deep_hidden, cnn_n, cnn_hidden):
        super(xDeepCNN, self).__init__()
        self.deep_n = deep_n
        self.deep_hidden = deep_hidden
        self.f = f
        self.k = k
        self.cnn_n = cnn_n
        self.cnn_hidden = cnn_hidden
        self.deeps = nn.ModuleList()
        self.deeps.append(nn.Linear(f*k, deep_hidden))
        for _i in range(1, deep_n):
            self.deeps.append(nn.Linear(deep_hidden, deep_hidden))

        self.cnns = nn.ModuleList()
        #         self.cnns.append(nn.Conv2d(self.f,self.cnn_hidden,(1,1)))
        for _i in range(0, self.cnn_n):
            self.cnns.append(nn.Conv2d(self.f,self.cnn_hidden,(1,1)))


    def _cross_cnn(self, x):  # (b, f, k)
        x_l_1 = x
        for conv in self.cnns:
            temp = x_l_1.unsqueeze(1) * x.unsqueeze(2) # (b,f,h,k)
            temp = conv(temp) # (b,h,h,k)
            temp = temp.sum(2)
            x_l_1 = temp
        del temp
        return x_l_1  # (b, hidden, k)

    def deep(self, x):
        x = x.view(x.size(0), -1)
        for m in self.deeps:
            x = m(x)
            x = F.relu(x)
        return x

    def forward(self, x):
        output1 = self._cross_cnn(x)
        output1 = output1.view(x.size(0), -1)
        output2 = self.deep(x)
        return torch.cat([output1, output2], 1)