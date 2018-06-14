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
from sklearn.model_selection import KFold, StratifiedKFold
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

    gbm = lgb_train_model(trainx, trainy, params, use_valid, valid_ratio, validx, validy,
                          num_boost_round, early_stopping_rounds, random_state)
    iteration = gbm.best_iteration
    if (iteration <= 0):
        iteration = num_boost_round
    return gbm.predict(testx, iteration)


def lgb_train_model(trainx, trainy, params, use_valid=True, valid_ratio=0.2, validx=None,
                    validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018):
    """
    lgb train 返回train的model
    :param trainx:
    :param trainy:
    :param params:
    :param use_valid:
    :param valid_ratio:
    :param validx:
    :param validy:
    :param num_boost_round:
    :param early_stopping_rounds:
    :param random_state:
    :return:
    """
    if use_valid:
        if (validx is None):
            trainx, validx, trainy, validy = train_test_split(trainx,
                                                              trainy,
                                                              test_size=valid_ratio,
                                                              random_state=random_state)
        lgb_train_ = lgb.Dataset(trainx, trainy)
        lgb_eval = lgb.Dataset(validx, validy, reference=lgb_train_)
        params_copy = params.copy()
        gbm = lgb.train(params_copy,
                        lgb_train_,
                        num_boost_round=num_boost_round,
                        valid_sets=[lgb_eval],
                        early_stopping_rounds=early_stopping_rounds)
    else:
        lgb_train_ = lgb.Dataset(trainx, trainy)
        params_copy = params.copy()
        gbm = lgb.train(params_copy,
                        lgb_train_,
                        num_boost_round=num_boost_round)
    return gbm


def lgb_cv(trainx, trainy, params, num_boost_round=500,
           use_valid=True, valid_ratio=0.2, early_stopping_rounds=5,
           cv=5, random_state=2018, func=mean_squared_error):
    label_size = _get_label_size(trainy)
    # cv
    final_score = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(trainx, trainy):
        scores = []
        if (label_size > 1):
            for i in range(label_size):
                pred = lgb_train(trainx[train_index], trainy[train_index][:, i],
                                 trainx[test_index], params, num_boost_round=num_boost_round,
                                 use_valid=use_valid, valid_ratio=valid_ratio,
                                 early_stopping_rounds=early_stopping_rounds)
                scores.append(func(trainy[test_index][:, i], pred))
            final_score.append(np.mean(scores))
        else:
            pred = lgb_train(trainx[train_index], trainy[train_index],
                             trainx[test_index], params, num_boost_round=num_boost_round,
                             use_valid=use_valid, valid_ratio=valid_ratio, early_stopping_rounds=early_stopping_rounds)
            final_score.append(func(trainy[test_index], pred))
    return final_score


def sklearn_cv(model, trainx, trainy, cv=5, random_state=2018, train_all_label=True, func=mean_squared_error):
    # cv
    label_size = _get_label_size(trainy)
    final_score = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(trainx, trainy):
        scores = []
        if (label_size > 1):
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


def sklearn_train(model, trainx, trainy, testx, preditc_proba=False):
    model = sklearn_train_model(model, trainx, trainy)
    if (preditc_proba):
        predict = model.predict_proba(testx)
    else:
        predict = model.predict(testx)
    return predict


def sklearn_train_model(model, trainx, trainy):
    model = clone(model)
    model.fit(trainx, trainy)
    return model


def lgb_predict(trainx, trainy, testx,
                params, use_valid=True, valid_ratio=0.2, validx=None,
                validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
                predict_prob=False):
    label_size = _get_label_size(trainy)
    if (label_size > 1):
        preds = []
        for i in range(label_size):
            pred = lgb_train(trainx, trainy[:, i], testx, params.copy(), use_valid, valid_ratio,
                             validx, validy, num_boost_round, early_stopping_rounds, random_state, predict_prob)
            preds.append(pred)
        return np.concatenate([preds]).T
    else:
        return lgb_train(trainx, trainy, testx, params.copy(), use_valid, valid_ratio,
                         validx, validy, num_boost_round, early_stopping_rounds, random_state, predict_prob)


def sklearn_predict(model, trainx, trainy, testx, train_all_label=True):
    label_size = _get_label_size(trainy)
    if train_all_label:
        return sklearn_train(model, trainx, trainy, testx)
    else:
        if (label_size > 1):
            predicts = []
            for i in range(5):
                predict = sklearn_train(model, trainx, trainy[:, i], testx)
                predicts.append(predict)
            return np.concatenate([predicts]).T
        else:
            return sklearn_train(model, trainx, trainy, testx)
