#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-9-21, 12:45

@Description:

@Update Date: 18-9-21, 12:45
"""
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        vocb_size = args['vocb_size']
        self.word_dim = args['dim']
        n_class = args['n_class']
        self.max_len = args['max_len']
        embedding_matrix = args['embedding_matrix']
        self.model_name = "TextCNN"
        # 需要将事先训练好的词向量载入
        self.embeding = nn.Embedding(vocb_size, self.word_dim)
        self.embeding.weight = embedding_matrix
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                          stride=1, padding=2),

                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)  # (16,64,64)
        )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2)
        )
        self.out = nn.Linear(512, n_class)

    def forward(self, x):
        x = self.embeding(x)
        x = x.view(x.size(0), 1, self.max_len, self.word_dim)
        # print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print(x.size())
        output = self.out(x)
        return output
