#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-5-12, 10:25

@Description:

@Update Date: 18-5-12, 10:25
"""

import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.base import clone

# lgb params
LGB_REGRESS_PARAMS = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'l2',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

LGB_BINARY_PARAMS = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'auc'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}


def _get_label_size(trainy):
    """
    检测是多label还是单label
    :param trainy:
    :return:
    """
    if (len(trainy.shape) > 1):
        label_size = trainy.shape[1]
    else:
        label_size = 1
    return label_size


def lgb_train(trainx, trainy, testx, params, use_valid=True, valid_ratio=0.2, validx=None,
              validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
              predict_prob=False):  # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    if use_valid:
        if (validx is None):
            trainx, validx, trainy, validy = train_test_split(trainx,
                                                              trainy,
                                                              test_size=valid_ratio,
                                                              random_state=random_state)
        lgb_train = lgb.Dataset(trainx, trainy)
        lgb_eval = lgb.Dataset(validx, validy, reference=lgb_train)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=num_boost_round,
                        valid_sets=[lgb_eval],
                        early_stopping_rounds=early_stopping_rounds)
        y_pred = gbm.predict(testx, num_iteration=gbm.best_iteration)
    else:
        lgb_train = lgb.Dataset(trainx, trainy)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=num_boost_round)
        y_pred = gbm.predict(testx)
    return y_pred


def lgb_cv(trainx, trainy, params,num_boost_round=500, cv=5, random_state=2018, func=mean_squared_error):
    label_size = _get_label_size(trainy)
    # cv
    final_score = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(trainx):
        scores = []
        if (label_size > 1):
            for i in range(label_size):
                pred = lgb_train(trainx[train_index], trainy[train_index][:, i],
                                 trainx[test_index], params, num_boost_round=num_boost_round)
                scores.append(func(trainy[test_index][:, i], pred))
            final_score.append(np.mean(scores))
        else:
            pred = lgb_train(trainx[train_index], trainy[train_index],
                             trainx[test_index], params, num_boost_round=num_boost_round)
            final_score.append(func(trainy[test_index], pred))
    return final_score


def sklearn_cv(model, trainx, trainy, cv=5, random_state=2018, train_all_label=True, func=mean_squared_error):
    # cv
    label_size = _get_label_size(trainy)
    final_score = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(trainx):
        scores = []
        if(label_size > 1):
            if train_all_label:
                pred = sklearn_train(model, trainx[train_index], trainy[train_index], trainx[test_index])
                for i in range(label_size):
                    scores.append(func(trainy[test_index][:, i], pred[:, i]))
            else:
                for i in range(label_size):
                    pred = sklearn_train(model, trainx[train_index], trainy[train_index][:, i],
                                         trainx[test_index])
                    scores.append(func(trainy[test_index][:, i], pred))
            final_score.append(scores)
        else:
            pred = sklearn_train(model, trainx[train_index], trainy[train_index], trainx[test_index])
            final_score.append(func(trainy[test_index], pred))
    return final_score


def sklearn_train(model, trainx, trainy, testx):
    model = clone(model)
    model.fit(trainx, trainy)
    predict = model.predict(testx)
    return predict


def lgb_predict(trainx, trainy, testx):
    label_size = _get_label_size(trainy)
    if (label_size > 1):
        preds = []
        for i in range(label_size):
            pred = lgb_train(trainx, trainy[:, i], testx)
            preds.append(pred)
        return np.concatenate([preds]).T
    else:
        return lgb_train(trainx, trainy, testx)


def sklearn_predict(model, trainx, trainy, testx, train_all_label=True):
    label_size = _get_label_size(trainy)
    if train_all_label:
        return sklearn_train(model, trainx, trainy, testx)
    else:
        if(label_size > 1):
            predicts = []
            for i in range(5):
                predict = sklearn_train(model, trainx, trainy[:, i], testx)
                predicts.append(predict)
            return np.concatenate([predicts]).T
        else:
            return sklearn_train(model, trainx, trainy, testx)