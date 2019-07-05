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
from joblib import dump, load, Parallel, delayed
# from smooth import HyperParam
import contextlib
from contextlib import contextmanager
from itertools import product
from DMF.smooth import HyperParam
import jieba
from multiprocessing import Pool
from functools import partial
import feather

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


def checkpath(f):
    # 检查一下用来存放checkpoint的path是否存在了
    @functools.wraps(f)
    def fn(*args, **kw):
        if 'checkpath' in args[0].keys():
            if os.path.exists(args[0]['checkpath']) == False:
                os.makedirs(args[0]['checkpath'])
        f(*args, **kw)

    return fn


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
            fname += ".feather"
            dump_path = os.path.join(path, fname)
        else:
            dump_path = os.path.join(path, f.__name__ + '.feather')
        if os.path.exists(dump_path):
            r = pd.read_feather(dump_path, nthreads=4)
        else:
            r = f(*args, **kw)
            downcast(r)
            r.reset_index(drop=True, inplace=True)
            for c in r.columns:
                if r[c].dtype == 'float64':
                    r[c] = r[c].astype('float32')
            r.to_feather(dump_path)
        gc.collect()
        t_end = time.time()
        print('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r
    return fn


MAIN_ID = ["uid", "pid"]
SORT_ID = ['_index']


def dump_feature_remove_main_id(f):  # 定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    @functools.wraps(f)
    def fn(*args, **kw):  # 对传进来的函数进行包装的函数
        path = os.path.join(args[-1], "features_remove_main_id")
        if (not os.path.exists(path)):
            os.mkdir(path)
        t_start = time.time()
        if (len(args) > 1):
            fname = f.__name__
            for _n in args[:-1]:
                fname += "_{}".format(_n)
            fname += ".feather"
            dump_path = os.path.join(path, fname)
        else:
            dump_path = os.path.join(path, f.__name__ + '.feather')
        if os.path.exists(dump_path):
            r = pd.read_feather(dump_path, nthreads=4)
        else:
            r = f(*args, **kw)
            r.sort_values(by=SORT_ID, inplace=True)
            # remove main id
            if (not f.__name__.startswith('click_label')):
                for _c in MAIN_ID:
                    if (_c in r.columns):
                        del r[_c]
                for _c in SORT_ID:
                    if (_c in r.columns):
                        del r[_c]
            # down bit
            for c in r.columns:
                if r[c].dtype == 'float64':
                    r[c] = r[c].astype('float32')
            r.reset_index(drop=True, inplace=True)
            downcast(r)
            r.to_feather(dump_path)
        gc.collect()
        t_end = time.time()
        print('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r
    return fn

def dump_feature_remove_main_id2(f):  # 定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    @functools.wraps(f)
    def fn(*args, **kw):  # 对传进来的函数进行包装的函数
        path = os.path.join(args[-1], "features_remove_main_id")
        if (not os.path.exists(path)):
            os.mkdir(path)
        t_start = time.time()
        if (len(args) > 1):
            fname = f.__name__
            for _n in args[:-1]:
                fname += "_{}".format(_n)
            fname += ".feather"
            dump_path = os.path.join(path, fname)
        else:
            dump_path = os.path.join(path, f.__name__ + '.feather')
        if os.path.exists(dump_path):
            r = feather.read_dataframe(dump_path)
        else:
            r = f(*args, **kw)
            r.sort_values(by=SORT_ID, inplace=True)
            # remove main id
            if (not f.__name__.startswith('click_label')):
                for _c in MAIN_ID:
                    if (_c in r.columns):
                        del r[_c]
                for _c in SORT_ID:
                    if (_c in r.columns):
                        del r[_c]
            # down bit
            for c in r.columns:
                if r[c].dtype == 'float64':
                    r[c] = r[c].astype('float32')
            r.reset_index(drop=True, inplace=True)
            downcast(r)
            r.to_feather(dump_path)
        gc.collect()
        t_end = time.time()
        print('call %s() in %fs' % (f.__name__, (t_end - t_start)))
        return r
    return fn


# 合并节约内存
@performance
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result


def log(labels):
    return np.log(labels + 1)


def exp(labels):
    return np.exp(labels) - 1

def getDegree(pointsA, pointsB):
    """
    Args:
        point pA(latA, lonA)
        point pB(latB, lonB)
        lat 0-90
        lon 0-180
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    latA = pointsA[:,0]
    lonA = pointsA[:,1]
    latB = pointsB[:,0]
    lonB = pointsB[:,1]
    radLatA = np.radians(latA)
    radLonA = np.radians(lonA)
    radLatB = np.radians(latB)
    radLonB = np.radians(lonB)
    dLon = radLonB - radLonA
    y = np.sin(dLon) * np.cos(radLatB)
    x = np.cos(radLatA) * np.sin(radLatB) - np.sin(radLatA) * np.cos(radLatB) * np.cos(dLon)
    brng = np.degrees(np.arctan2(y, x))
    brng = (brng + 360) % 360
    return brng

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
    余弦距离
    :param values1: n_sample * f_leangth
    :param values2: n_sample * f_leangth 或者 f_leangth
    :return:
    """
    return np.sum((values1 * values2), axis=1) / (np.sqrt(np.sum(values1 ** 2, axis=1)) * np.sqrt(np.sum(values2 ** 2)))


MAX_DIS_FUNCTION.append(cosine)


def optm_cosine(v1, v2):
    """
    优化一波余弦距离
    """
    n_samples, f_features = v1.shape
    n_centers, f_features = v2.shape
    return np.dot(v1, v2.T) / np.dot(np.linalg.norm(v1, axis=1).reshape(n_samples, -1),
                                     np.linalg.norm(v2, axis=1).reshape(-1, n_centers))


MAX_DIS_FUNCTION.append(optm_cosine)


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


def fill_feature(df, fill_features, cate_method="mode", num_method="median", other_fill_value=-1, threshold=0.01):
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
            if (df[_f].dtypes == object):
                if (cate_method == "mode"):
                    df[_f] = df[_f].fillna(df[_f].mode()[0])
                else:
                    print("error fill cate method", num_method)
                    exit(1)
            else:
                if (num_method == "median"):
                    df[_f] = df[_f].fillna(df[_f].median())
                elif (num_method == "mean"):
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


def detect_cates_for_narrayx(x, threshold=10):
    """
    将数量小于等于threshold的认为 是cate列
    :param x: np_array
    :param threshold:
    :return:
    """
    cates = []
    for _i in range(x.shape[1]):
        if (len(np.unique(x[:, _i])) <= threshold):
            if (np.isnan(x[:, _i]).sum() == 0):
                x[:, _i] = x[:, _i].astype(int)
                cates.append(_i)
    return cates


def transform_float_to_int_for_narrayx(x, cates):
    """
    将x中cates的列的全部转换为int
    :param x: np_array
    :param cates:
    :return:
    """
    x = x.copy()
    for _i in cates:
        x[:, _i] = x[:, _i].astype(int)
    return x

@performance
def downcast(df):
    """
    降位
    :param df:
    :return:
    """
    print(df.info(memory_usage='deep'))
    df_int = df.select_dtypes(include=['int64', 'int32']).apply(pd.to_numeric, downcast='integer')
    df[df_int.columns] = df_int
    del df_int;
    gc.collect()
    print(df.info(memory_usage='deep'))
    df_float64 = df.select_dtypes(include=['float64']).apply(pd.to_numeric, downcast='float')
    df[df_float64.columns] = df_float64
    del df_float64;
    gc.collect()
    print(df.info(memory_usage='deep'))


def lgb_stacking_feature(params, trainx, trainy, testx, probe_name, topk=0, feature_names=None, cv=3, rounds=3):
    from DMF.Stacking import StackingBaseModel
    newtrain = np.zeros(shape=(trainx.shape[0],))
    newtest = np.zeros(shape=(testx.shape[0],))

    for _i in range(rounds):
        stack = StackingBaseModel(None, "lgb", params, cv, use_valid=False, random_state=2018 * _i,
                                  top_k_origin_feature=topk)
        stack.set_feature_names(feature_names)
        _ntest = stack.fit_transfrom(trainx, trainy, testx)
        _ntrain = stack.trainx_
        newtrain += _ntrain[:, 0]
        newtest += _ntest[:, 0]
    newtrain /= rounds
    newtest /= rounds
    topkfname = stack.topk_feature_name
    dftrain = pd.DataFrame(newtrain)
    dftest = pd.DataFrame(newtest)
    dftrain.columns = [probe_name + "_probe"]
    dftest.columns = [probe_name + "_probe"]
    df1 = pd.concat([dftrain, dftest], axis=0).reset_index(drop=True)
    if (topkfname is not None):
        newtrain2 = _ntrain[:, 1:]
        newtest2 = _ntest[:, 1:]
        dftrain2 = pd.DataFrame(newtrain2)
        dftest2 = pd.DataFrame(newtest2)
        dftrain2.columns = topkfname
        dftest2.columns = topkfname
        df2 = pd.concat([dftrain2, dftest2], axis=0).reset_index(drop=True)
        df1 = concat([df1, df2])
    return df1


def xgb_stacking_feature(params, trainx, trainy, testx, probe_name, topk=0, feature_names=None, cv=3, rounds=3):
    from DMF.Stacking import StackingBaseModel
    newtrain = np.zeros(shape=(trainx.shape[0],))
    newtest = np.zeros(shape=(testx.shape[0],))

    for _i in range(rounds):
        stack = StackingBaseModel(None, "xgb", params, cv, use_valid=False, random_state=2018 * _i,
                                  top_k_origin_feature=topk)
        stack.set_feature_names(feature_names)
        _ntest = stack.fit_transfrom(trainx, trainy, testx)
        _ntrain = stack.trainx_
        newtrain += _ntrain[:, 0]
        newtest += _ntest[:, 0]
    newtrain /= rounds
    newtest /= rounds
    topkfname = stack.topk_feature_name
    dftrain = pd.DataFrame(newtrain)
    dftest = pd.DataFrame(newtest)
    dftrain.columns = [probe_name + "_probe"]
    dftest.columns = [probe_name + "_probe"]
    df1 = pd.concat([dftrain, dftest], axis=0).reset_index(drop=True)
    if (topkfname is not None):
        newtrain2 = _ntrain[:, 1:]
        newtest2 = _ntest[:, 1:]
        dftrain2 = pd.DataFrame(newtrain2)
        dftest2 = pd.DataFrame(newtest2)
        dftrain2.columns = topkfname
        dftest2.columns = topkfname
        df2 = pd.concat([dftrain2, dftest2], axis=0).reset_index(drop=True)
        df1 = concat([df1, df2])
    return df1


def catboost_stacking_feature(model, trainx, trainy, testx, probe_name, topk=0, feature_names=None, cv=3, rounds=3):
    from DMF.Stacking import StackingBaseModel
    newtrain = np.zeros(shape=(trainx.shape[0],))
    newtest = np.zeros(shape=(testx.shape[0],))

    for _i in range(rounds):
        stack = StackingBaseModel(model, "catboost", None, cv, use_valid=False, random_state=2018 * _i,
                                  top_k_origin_feature=topk)
        stack.set_feature_names(feature_names)
        _ntest = stack.fit_transfrom(trainx, trainy, testx)
        _ntrain = stack.trainx_
        newtrain += _ntrain[:, 0]
        newtest += _ntest[:, 0]
    newtrain /= rounds
    newtest /= rounds
    topkfname = stack.topk_feature_name
    dftrain = pd.DataFrame(newtrain)
    dftest = pd.DataFrame(newtest)
    dftrain.columns = [probe_name + "_probe"]
    dftest.columns = [probe_name + "_probe"]
    df1 = pd.concat([dftrain, dftest], axis=0).reset_index(drop=True)
    if (topkfname is not None):
        newtrain2 = _ntrain[:, 1:]
        newtest2 = _ntest[:, 1:]
        dftrain2 = pd.DataFrame(newtrain2)
        dftest2 = pd.DataFrame(newtest2)
        dftrain2.columns = topkfname
        dftest2.columns = topkfname
        df2 = pd.concat([dftrain2, dftest2], axis=0).reset_index(drop=True)
        df1 = concat([df1, df2])
    return df1


def sklearn_stacking_feature(model, trainx, trainy, testx, probe_name, topk=0, feature_names=None, cv=3, rounds=3):
    from DMF.Stacking import StackingBaseModel
    newtrain = np.zeros(shape=(trainx.shape[0],))
    newtest = np.zeros(shape=(testx.shape[0],))

    for _i in range(rounds):
        stack = StackingBaseModel(model, "sklearn", None, cv, use_valid=False, random_state=2018 * _i,
                                  top_k_origin_feature=topk)
        stack.set_feature_names(feature_names)
        _ntest = stack.fit_transfrom(trainx, trainy, testx)
        _ntrain = stack.trainx_
        newtrain += _ntrain[:, 0]
        newtest += _ntest[:, 0]
    newtrain /= rounds
    newtest /= rounds
    topkfname = stack.topk_feature_name
    dftrain = pd.DataFrame(newtrain)
    dftest = pd.DataFrame(newtest)
    dftrain.columns = [probe_name + "_probe"]
    dftest.columns = [probe_name + "_probe"]
    df1 = pd.concat([dftrain, dftest], axis=0).reset_index(drop=True)
    if (topkfname is not None):
        newtrain2 = _ntrain[:, 1:]
        newtest2 = _ntest[:, 1:]
        dftrain2 = pd.DataFrame(newtrain2)
        dftest2 = pd.DataFrame(newtest2)
        dftrain2.columns = topkfname
        dftest2.columns = topkfname
        df2 = pd.concat([dftrain2, dftest2], axis=0).reset_index(drop=True)
        df1 = concat([df1, df2])
    return df1


def encode_vt(train_df, test_df, variable, target, use_bayes=True):
    if (type(variable) != list):
        variable = [variable]
    variable = list(variable)
    col_name = "_".join(variable)
    col_name = col_name + "_" + target
    grouped = train_df.groupby(variable, as_index=False)[target].agg({"C": "size", "V": "sum"})
    C = grouped['C']
    V = grouped['V']
    if(use_bayes):
        print('start smooth')
        hyper = HyperParam(1, 1)
        hyper.update_from_data(C, V)
        print('end smooth')
        grouped[col_name] = (hyper.alpha + V) / (hyper.alpha + hyper.beta + C)
    else:
        grouped[col_name] = V / C
    grouped[col_name] = grouped[col_name].astype('float32')
    df = test_df[variable].merge(grouped, 'left', variable)[col_name]
    df = np.asarray(df, dtype=np.float32)
    return df

@performance
def transform(id_cate, target, train, test, use_bayes_smooth=True, bayes_n_split=5):
    if (type(id_cate) != list):
        id_cate = [id_cate]
    print("%s unique num: %s" % (id_cate, train[id_cate].nunique()))
    col_name = "_".join(list(id_cate))
    col_name = col_name + "_" + target + "_ctr"
    bayes_feature = encode_vt(train, test, id_cate, target, use_bayes_smooth)
    test[col_name] = bayes_feature
    skf = StratifiedKFold(n_splits=bayes_n_split, shuffle=True, random_state=2018)
    for i, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(train)), train[target])):
        print(id_cate, target, i)
        X_train = train.iloc[train_idx]
        X_test = train.iloc[test_idx]
        bayes_feature = encode_vt(X_train, X_test, id_cate, target, use_bayes_smooth)
        train.ix[train.iloc[test_idx].index, col_name] = bayes_feature
    return train[[col_name]], test[[col_name]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
    return -np.log(1/x - 1)

def sigmoid_avg(xs):
    newxs = []
    for _x in xs:
        newxs.append(inverse_sigmoid(_x))
    x = np.mean(newxs,axis=0)
    return sigmoid(x)

def rank_avg(ress, weightss, label_name, id_name):
    for a in ress:
        a.sort_values(by=label_name,inplace=True)
        a['rank'] = np.arange(a.shape[0]) / a.shape[0]
        a.sort_values(id_name,inplace=True)
    c = a.copy()
    c[label_name] = 0
    for _w, _a in zip(weightss,ress):
        c[label_name] = c[label_name].values + (_a['rank'].values * _w)
    del c['rank']
    return c



def count_ratio(func, func2, click_label_func, dataPath):
    """

    :param func: specify_count_df_func
    :param func2: count_df_func
    :param click_label_func:
    :param dataPath:
    :return:
    """
    a1 = func(dataPath)
    a2 = func2(dataPath)
    a3 = click_label_func(dataPath)
    res = a3[['_index']]
    for _c in a1.columns.tolist():
        res[_c + "_ratio"] = a1[_c] / a2[a2.columns[0]]
    return res

def mkpath(path):
    if (not os.path.exists(path)):
        os.mkdir(path)


#并行 map更快
def map_func2(func, data, n_jobs):
    agents = n_jobs
    chunksize = len(data) // n_jobs + 1
    with contextlib.closing(Pool(processes=agents)) as pool:
        res = pool.map(func, data, chunksize)
    return res
    
def map_func(func, data, n_thread = 8, chunksize=32):
    agents = n_thread
    with contextlib.closing(Pool(processes=agents)) as pool:
        res = pool.map(func, data, chunksize)
    return res

def apply_func(func, data, n_jobs):
    res = Parallel(n_jobs=n_jobs)(delayed(func)(x) for x in data)
    return res

# lgb进行特征选择
def lgb_important_features(dataPath, click_label_fun, NEED_VALID_FUN_set, feature_functions, lgb_params, num_boost_round = 300, topk = 30):
    from DMF.train_predict import lgb_train
    valid = False
    feature_names = []
    features = []
    for _fun in feature_functions:
        if (_fun in NEED_VALID_FUN_set):
            temp2 = _fun(valid, dataPath)
        else:
            temp2 = _fun(dataPath)
        features.append(temp2)
    features2 = concat(features)
    del temp2
    del features
    features = features2
    del features2
    feature_names += list(features.columns)
    temp = click_label_fun(dataPath)
    features[temp.columns.tolist()] = temp
    del temp
    gc.collect()
    temp = features
    del features
    gc.collect()
    downcast(temp)
    if '_index' in feature_names:
        feature_names.remove('_index')
    print(feature_names)
    print(len(feature_names))
    print("start train")
    df = temp
    train = df[df.Tag != -1]
    test = df[df.Tag == -1]
    trainx = train[feature_names].values
    testx = test[feature_names].values
    trainy = train['Tag'].values
    feature_importances = []
    lgb_train(trainx, trainy, testx, lgb_params, False, num_boost_round=num_boost_round,
                     early_stopping_rounds=0, feature_importances=feature_importances,
                     feature_names=feature_names)
    feature_importance_names = []

    for _i, _f in enumerate(feature_importances):
        if(_i < topk):
            feature_importance_names.append(_f[0])
        else:
            break
    return temp[feature_importance_names + ['_index']]

@performance
def reduce_mem_usage(df, verbose=True):
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [ (i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch) ]

def batch_generator1(X, batch_size, trn_idx):
    sample_size = len(trn_idx)
    index_array = np.arange(sample_size)
    batches = make_batches(sample_size, batch_size)
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        print (batch_index*batch_size)
        batch_ids = index_array[batch_start:batch_end]
        yield X.iloc[np.array(trn_idx)[batch_ids]].values

@performance
def to_bin(x,file,idx):
    print('to bin', file)
    train_np = np.array([], dtype=np.float32)
    count = 0
    for chunk in batch_generator1(x, 2000000, idx):
        temp = np.array(chunk.reshape(-1), dtype=np.float32)
        train_np = np.append(train_np, temp)
        del temp; gc.collect()
        count += chunk.shape[0]
        del chunk; gc.collect()
        print ("#", count)
    train_np.tofile(file)

@performance
def set_newid(alldata, cate_ids):
    for _c in cate_ids:
        alldata[_c] = alldata[_c].astype(str)
        _interid = np.intersect1d(alldata[alldata.type == 'test'][_c], alldata[alldata.type!='test'][_c])
        _d = dict(zip(_interid,range(len(_interid))))
        alldata[_c + '_newid'] = alldata[_c].apply(lambda x: _d.setdefault(x,-1))
        alldata[_c + '_newid'] = alldata[_c + '_newid'].astype(int)
    return alldata

def make_bin(make_bin_df, col, min_sample_in_bin):
    temp = make_bin_df.groupby(col,as_index=False)[col].agg({'c':'count'})
    temp = temp.sort_values(by=col)
    _dict = {}
    _prec = 0
    _newid = 0
    for _k,_v in temp[[col,'c']].values:
        _prec += _v
        if(_prec >= min_sample_in_bin):
            _newid += 1
            _prec = 0
        _dict[_k] = _newid    
    return make_bin_df[col].map(_dict).fillna(0).astype(int).values

@performance
def make_bins(f, cols, min_sample_in_bin):
    global make_bin_df
    make_bin_df = f
    fs = []
    for col in cols:
        print(col, 'to bin')
        f[col + '_bin'] = make_bin(f[[col]],col,min_sample_in_bin)
        fs.append(col + '_bin')
    return f[fs]

make_bin_df = None

def make_bin_parallel(col, min_sample_in_bin):
    temp = make_bin_df.groupby(col,as_index=False)[col].agg({'c':'count'})
    temp = temp.sort_values(by=col)
    _dict = {}
    _prec = 0
    _newid = 0
    for _k,_v in temp[[col,'c']].values:
        _prec += _v
        if(_prec >= min_sample_in_bin):
            _newid += 1
            _prec = 0
        _dict[_k] = _newid    
    return make_bin_df[col].map(_dict).fillna(0).astype(int).values

@performance
def make_bins_parallel(f, cols, min_sample_in_bin,n_thread = 8):
    global make_bin_df
    make_bin_df = f
    fs = []
    for col in cols:
        print(col, 'to bin')
        fs.append(col + '_bin')
    ress = map_func(partial(make_bin_parallel,min_sample_in_bin = min_sample_in_bin),cols,n_thread)  
    res = pd.DataFrame()
    for _f,_r in zip(fs,ress):
        res[_f] = _r
    return res


if __name__ == '__main__':

    print(cosine(np.asarray([[1, 2, 3], [2, 5, 10], [2, 4, 6]]), np.asarray([1, 2, 3])))
    print(euclidean.__name__)
    print(cosine in MAX_DIS_FUNCTION)
