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
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder



MAX_DIS_FUNCTION = []


################## 辅助函数 #####################
def performance(f):  # 定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    @functools.wraps(f)
    def fn(*args, **kw):  # 对传进来的函数进行包装的函数
        t_start = time.time()  # 记录函数开始时间
        r = f(*args, **kw)  # 调用函数
        t_end = time.time()  # 记录函数结束时间
        print('call %s() in %fs' % (f.__name__, (t_end - t_start)))  # 打印调用函数的属性信息，并打印调用函数所用的时间
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
        print('call %s() in %fs' % (f.__name__, (t_end - t_start)))
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
def filter_feature_by_missing(df, origin_feature_names, threshold=0.95):
    """
    过滤缺失率大于threshold的特征
    :param df: 数据集df
    :param origin_feature_names: 要过滤的特征
    :param threshold:
    :return: 去除特征，保留特征
    """
    remove_feature_name = []
    stay_feature_name = []
    for _f in origin_feature_names:
        nan_size = df[_f].isnull().sum()
        nan_ratio = float(nan_size) / df.shape[0]
        if nan_ratio > threshold:
            remove_feature_name.append(_f)
        else:
            stay_feature_name.append(_f)
    return remove_feature_name, stay_feature_name


def filter_feature_by_spearmanr(df, origin_feature_names, target_column, threshold=0.1):
    """
    过滤皮尔森相关系数小于threshold的特征
    :param df:
    :param origin_feature_names:
    :param threshold:
    :return:
    """
    remove_feature_name = []
    stay_feature_name = []
    y = df[target_column].values
    for _f in origin_feature_names:
        if (df[_f].dtypes == object):
            temp = label_encode(df[[_f]].copy(), [_f])[_f]
        else:
            temp = df[_f]
        score = spearmanr(temp.values, y).correlation
        if score < threshold:
            remove_feature_name.append(_f)
        else:
            stay_feature_name.append(_f)
    return remove_feature_name, stay_feature_name


def filter_feature_which_0std(df, origin_feature_names):
    """
    去除方差为0的特征列
    :param df:
    :param origin_feature_names:
    :return:
    """
    remove_feature_name = []
    stay_feature_name = []
    for _f in origin_feature_names:
        if (df[_f].dtypes == object):
            temp = df[_f]
            if (len(temp[~temp.isnull()].unique()) <= 1):
                remove_feature_name.append(_f)
            else:
                stay_feature_name.append(_f)
        else:
            s = df[_f].std()
            if s == 0:
                remove_feature_name.append(_f)
            else:
                stay_feature_name.append(_f)
    return remove_feature_name, stay_feature_name

def fill_feature(df, fill_features, cate_method="mode", num_method="median", other_fill_value=-1, threshold = 0.01):
    """
    填充缺失率小于threshold的列，mode 为众数填充，median为中位数填充，mean为均值填充
    超过threshold的填充-1
    :param df:
    :param fill_features:
    :param cate_method:
    :param num_method:
    :param threshold:
    :return:
    """
    for _f in fill_features:
        nan_size = df[_f].isnull().sum()
        nan_ratio = float(nan_size) / df.shape[0]
        if nan_ratio < threshold:
            if(df[_f].dtypes == object):
                if(cate_method == "mode"):
                    df[_f] = df[_f].fillna(df[_f].mode()[0])
                else:
                    print("error fill cate method", num_method)
                    exit(1)
            else:
                if(num_method == "median"):
                    df[_f] = df[_f].fillna(df[_f].median())
                elif(num_method == "mean"):
                    df[_f] = df[_f].fillna(df[_f].mean())
                else:
                    print("error fill num method", num_method)
                    exit(1)
        else:
            df[_f] = df[_f].fillna(other_fill_value)

    return df

def detect_cate_columns(df, detect_columns):
    """
    检测cate列和数字列
    :param df:
    :param detect_columns:
    :return: 类别类,数字列
    """
    part = df[detect_columns]
    cate_columns = part.columns[part.dtypes == object]
    num_columns = part.columns[part.dtypes != object]
    return cate_columns, num_columns


def label_encode(df, transform_columns):
    """
    将df中的transform_columns列转为label列
    :param df:
    :param transform_columns:
    :return:
    """
    for _c in transform_columns:
        uniques = df[_c].unique()
        _d = dict(zip(uniques, range(len(uniques))))
        df[_c] = df[_c].apply(lambda x: _d[x])
    return df


def train_test_split_stratifiedKFold(n_split, random_state, shuffle, target_df, target_column):
    """
    StratifiedKFold划分训练和验证
    :param n_split:
    :param random_state:
    :param shuffle:
    :param target_df:
    :param target_column:
    :return: [(train1_index,valid1_index),(train2_index,valid2_index),...]
    """
    skf = StratifiedKFold(n_splits=n_split, random_state=random_state, shuffle=shuffle)
    g = skf.split(np.arange(target_df.shape[0]), target_df[target_column].values)
    splits = []
    for _x, _y in g:
        splits.append((_x, _y))
    return splits


if __name__ == '__main__':
    print(cosine(np.asarray([[1, 2, 3], [2, 5, 10], [2, 4, 6]]), np.asarray([1, 2, 3])))
    print(euclidean.__name__)
    print(cosine in MAX_DIS_FUNCTION)
