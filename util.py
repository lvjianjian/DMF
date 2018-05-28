#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-5-12, 10:24

@Description:

@Update Date: 18-5-12, 10:24
"""
import numpy as np
import time
import os
import gc
import pandas as pd
from sklearn.metrics import roc_auc_score
import functools

MAX_DIS_FUNCTION = []

################## 辅助函数 #####################
def performance(f):  # 定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    @functools.wraps(f)
    def fn(*args, **kw):  # 对传进来的函数进行包装的函数
        t_start = time.time()  # 记录函数开始时间
        r = f(*args, **kw)  # 调用函数
        t_end = time.time()  # 记录函数结束时间
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start)))  # 打印调用函数的属性信息，并打印调用函数所用的时间
        return r  # 返回包装后的函数

    return fn  # 调用包装后的函数


def dump_feature(f):  # 定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    @functools.wraps(f)
    def fn(*args, **kw):  # 对传进来的函数进行包装的函数
        path = os.path.join(args[-1], "features")
        if (not os.path.exists(path)):
            os.mkdir(path)
        t_start = time.time()
        if (len(args) > 1):
            fname = f.__name__
            for _n in args[:-1]:
                fname += "_{}".format(_n)
            fname += ".pickle"
            dump_path = os.path.join(path, fname)
        else:
            dump_path = os.path.join(path, f.__name__ + '.pickle')
        if os.path.exists(dump_path):
            r = pd.read_pickle(dump_path)
        else:
            r = f(*args, **kw)
            r.to_pickle(dump_path)
        gc.collect()
        t_end = time.time()
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r
    return fn


def log(labels):
    return np.log(labels + 1)


def exp(labels):
    return np.exp(labels) - 1



def euclidean(values1, values2):
    """
    欧式距离
    :param values1: n_sample * f_leangth
    :param values2: n_sample * f_leangth 或者 f_leangth
    :return:
    """
    return np.sqrt(np.sum((values1 - values2) ** 2, axis=1))

def cosine(values1, values2):
    """
    欧式距离
    :param values1: n_sample * f_leangth
    :param values2: n_sample * f_leangth 或者 f_leangth
    :return:
    """
    return np.sum((values1 * values2), axis=1) / (np.sqrt(np.sum(values1 ** 2, axis=1)) * np.sqrt(np.sum(values2 ** 2)))
MAX_DIS_FUNCTION.append(cosine)


# 过滤缺失值过多的特征
def filter_feature(train_df, origin_feature_names, threshold=0.95):
    """
    过滤缺失值大于threshold的特征
    :param train_df: 数据集df
    :param origin_feature_names: 要过滤的特征
    :param threshold:
    :return: 去除特征，保留特征
    """
    remove_feature_name = []
    stay_feature_name = []
    for _f in origin_feature_names:
        nan_size = train_df[_f].isnull().sum()
        nan_ratio = float(nan_size) / train_df.shape[0]
        if nan_ratio > threshold:
            remove_feature_name.append(_f)
        else:
            stay_feature_name.append(_f)
    return remove_feature_name, stay_feature_name


if __name__ == '__main__':
    print(cosine(np.asarray([[1, 2, 3], [2, 5, 10], [2, 4, 6]]), np.asarray([1, 2, 3])))
    print (euclidean.__name__)
    print (cosine in MAX_DIS_FUNCTION)