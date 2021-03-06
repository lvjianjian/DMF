#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-5-12, 10:25

@Description:

@Update Date: 18-5-12, 10:25
"""

import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from DMF.util import detect_cates_for_narrayx, transform_float_to_int_for_narrayx, performance, to_bin
import gc


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
              predict_prob=True, feature_importances=None, group=None, model_save_file=None,
              feature_names="auto", isFromFile=False, FileName=None,categorical_features='auto',
              eval_testx = None, trainx_weight = None, feval=None):  # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    gbm = lgb_train_model(trainx, trainy, params, use_valid, valid_ratio, validx, validy,
                          num_boost_round, early_stopping_rounds, random_state, feature_names, group,
                          isFromFile=isFromFile, FileName=FileName,categorical_features=categorical_features,
                          trainx_weight = trainx_weight,feval=feval)
    if (model_save_file is not None):
        gbm.save_model(model_save_file, num_iteration=gbm.best_iteration)
    if (type(feature_importances) == list):
        names, importances = zip(*(sorted(zip(gbm.feature_name(), gbm.feature_importance(importance_type='gain')), key=lambda x: -x[1])))
        feature_importances += list((zip(names, importances)))
        feature_importances.append(('best_score',gbm.best_score))
    if(eval_testx is None):
        return predict(gbm, "lgb", testx, predict_proba=predict_prob)
    else:
        return predict(gbm, "lgb", testx, predict_proba=predict_prob), predict(gbm, "lgb", eval_testx, predict_proba=predict_prob)


@performance
def lgb_train_model(trainx, trainy, params, use_valid=True, valid_ratio=0.2, validx=None,
                    validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
                    feature_names=None, group=None, isFromFile=False,FileName=None,categorical_features='auto',
                    trainx_weight = None,feval=None):
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
    deltrainx = False
    if (feature_names is None):
        feature_names = "auto"
    if (isFromFile):
        num_sample, num_feature = trainx.shape
        if(FileName is None):
            if (type(trainx) is pd.DataFrame):
                feature_names = list(trainx.columns)
            if (type(feature_names) is not list and type(feature_names) is not pd.Index and feature_names == "auto"):
                print("please set feature names when isFromFile == True")
                exit(1)

            print('shape', trainx.shape)
            # if(not os.path.exists('./X_train_cache.bin')):
            FileName = './X_train_cache.bin'
            to_bin(trainx, FileName, np.arange(trainx.shape[0]))
            del trainx
            deltrainx = True

    if use_valid:
        if (validx is None):
            if(trainx_weight is not None):
                print('trainx_weight is not None and use valid, exit')
                exit(1)
            trainx, validx, trainy, validy = train_test_split(trainx,
                                                              trainy,
                                                              test_size=valid_ratio,
                                                              random_state=random_state)
        if (isFromFile):
            lgb_train_ = lgb.Dataset(FileName,
                                     label=trainy,
                                     feature_name=feature_names,
                                     isFromFile=True,
                                     shape=(num_sample, num_feature),
                                    categorical_feature=categorical_features)
        else:
            lgb_train_ = lgb.Dataset(trainx, trainy, feature_name=feature_names,
                                     categorical_feature=categorical_features,weight=trainx_weight)
        if(not deltrainx):
            del trainx
        del trainy
        gc.collect()
        lgb_eval = lgb.Dataset(validx, validy, reference=lgb_train_,
                               categorical_feature=categorical_features)
        del validx
        del validy
        gc.collect()
        params_copy = params.copy()
        gbm = lgb.train(params_copy,
                        lgb_train_,
                        num_boost_round=num_boost_round*100,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=int(early_stopping_rounds / 10),
                       feval=feval)
    else:
        if (isFromFile):
            lgb_train_ = lgb.Dataset('./X_train_cache.bin',
                                     label=trainy,
                                     feature_name=feature_names,
                                     isFromFile=True,
                                     shape=(num_sample, num_feature),
                                     categorical_feature=categorical_features,
                                     weight=trainx_weight)
        else:
            lgb_train_ = lgb.Dataset(trainx, trainy, feature_name=feature_names, group=group,
                                     categorical_feature=categorical_features, weight=trainx_weight)
        if(not deltrainx):
            del trainx
        del trainy
        gc.collect()
        params_copy = params.copy()
        gbm = lgb.train(params_copy,
                        lgb_train_,
                        num_boost_round=num_boost_round)
    return gbm


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


def xgb_train(trainx, trainy, testx, params, use_valid=True, valid_ratio=0.2, validx=None,
              validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
              predict_prob=True, feature_importances=None, model_save_file=None,
              feature_names=None,eval_testx = None,feval=None):  # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    gbm = xgb_train_model(trainx, trainy, params, use_valid, valid_ratio, validx, validy,
                          num_boost_round, early_stopping_rounds, random_state, feature_names,feval=feval)
    if (model_save_file is not None):
        gbm.save_model(model_save_file)
    if (type(feature_importances) == list):
        names, importances = zip(*(sorted(gbm.get_fscore().items(), key=lambda x: -x[1])))
        feature_importances += list((zip(names, importances)))
    if(eval_testx is None):
        return predict(gbm, "xgb", testx, predict_proba=predict_prob,feature_names=feature_names)
    else:
        return predict(gbm, "xgb", testx, predict_proba=predict_prob,feature_names=feature_names),\
               predict(gbm, "xgb", eval_testx, predict_proba=predict_prob,feature_names=feature_names)

def xgb_train_model(trainx, trainy, params, use_valid=True, valid_ratio=0.2, validx=None,
                    validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
                    feature_names=None,feval=None):
    """
    xgb train 返回train的model
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
        xgb_train_ = xgb.DMatrix(trainx, trainy, feature_names=feature_names)
        del trainx
        del trainy
        gc.collect()
        xgb_eval = xgb.DMatrix(validx, validy, feature_names=feature_names)
        del validx
        del validy
        gc.collect()
        params_copy = params.copy()
        gbm = xgb.train(params_copy,
                        xgb_train_,
                        num_boost_round=num_boost_round,
                        evals=[(xgb_eval,'valid')],
                        early_stopping_rounds=early_stopping_rounds,feval=feval)
    else:
        xgb_train_ = xgb.DMatrix(trainx, trainy, feature_names=feature_names)
        del trainx
        del trainy
        gc.collect()
        params_copy = params.copy()
        gbm = xgb.train(params_copy,
                        xgb_train_,
                        num_boost_round=num_boost_round)
    return gbm


def xgb_predict(trainx, trainy, testx,
                params, use_valid=True, valid_ratio=0.2, validx=None,
                validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
                predict_prob=False):
    label_size = _get_label_size(trainy)
    if (label_size > 1):
        preds = []
        for i in range(label_size):
            pred = xgb_train(trainx, trainy[:, i], testx, params.copy(), use_valid, valid_ratio,
                             validx, validy, num_boost_round, early_stopping_rounds, random_state, predict_prob)
            preds.append(pred)
        return np.concatenate([preds]).T
    else:
        return xgb_train(trainx, trainy, testx, params.copy(), use_valid, valid_ratio,
                         validx, validy, num_boost_round, early_stopping_rounds, random_state, predict_prob)


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


def sklearn_train(model, trainx, trainy, testx, predict_proba=False,eval_testx = None):
    model = sklearn_train_model(model, trainx, trainy)
    if(eval_testx is None):
        return predict(model, model.__class__.__name__, testx, predict_proba)
    else:
        return predict(model, model.__class__.__name__, testx, predict_proba), \
               predict(model, model.__class__.__name__, eval_testx, predict_proba)


def sklearn_train_model(model, trainx, trainy):
    model = clone(model)
    model.fit(trainx, trainy)
    return model


def catboost_train_model(model, trainx, trainy, cate_threshold=10,cates =None):
    model = clone(model)
    if(cates is None):
        cates = detect_cates_for_narrayx(trainx, cate_threshold)
        trainx = transform_float_to_int_for_narrayx(trainx, cates)
    model.fit(trainx, trainy, cates)
    model.cates = cates
    return model


def catboost_train(model, trainx, trainy, testx, cate_threshold=10, predict_proba=True,eval_testx = None,cates=None):
    model = catboost_train_model(model, trainx, trainy, cate_threshold,cates)
    if(cates is None):
        testx = transform_float_to_int_for_narrayx(testx, model.cates)
    if(eval_testx is None):
        return predict(model, "catboost", testx, predict_proba)
    else:
        return predict(model, "catboost", testx, predict_proba), predict(model, "catboost", eval_testx, predict_proba)


def predict(trained_model, model_name, testx, predict_proba=True, feature_names=None):
    if (model_name == "lgb"):
        iteration = trained_model.best_iteration
        if (iteration <= 0):
            iteration = trained_model.current_iteration()
        if (iteration < 0):
            print("please set iteration, now it is {}".format(iteration))
            exit(1)
        if (predict_proba):
            return trained_model.predict(testx, iteration)
    elif (model_name == 'xgb'):
        testDM = xgb.DMatrix(testx, feature_names=feature_names)
        return trained_model.predict(testDM)
    else:
        if (predict_proba):
            return trained_model.predict_proba(testx)
        else:
            return trained_model.predict(testx)

# 通用train和predict方法
def general_train(model_type, trainx, trainy, testx, params, use_valid=True, valid_ratio=0.2, validx=None,
                  validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
                  predict_prob=True, feature_importances=None, group=None, model_save_file=None,
                  feature_names="auto", isFromFile=False, model=None, cate_threshold=10, categorical_features='auto',
                  eval_testx = None, trainx_weight = None, feval=None):
    if (model_type == 'lgb'):
        return lgb_train(trainx, trainy, testx, params, use_valid, valid_ratio, validx, validy,
                         num_boost_round, early_stopping_rounds, random_state, predict_prob,
                         feature_importances, group, model_save_file, feature_names, isFromFile,
                        categorical_features=categorical_features,eval_testx=eval_testx, trainx_weight = trainx_weight, feval=feval)
    elif (model_type == 'xgb'):
        return xgb_train(trainx, trainy, testx, params, use_valid, valid_ratio, validx, validy,
                         num_boost_round, early_stopping_rounds, random_state, predict_prob,
                         feature_importances, model_save_file, feature_names,eval_testx=eval_testx,feval=feval)
    elif (model_type == 'catboost'):
        return catboost_train(model, trainx, trainy, testx, cate_threshold,eval_testx=eval_testx,predict_proba=predict_prob,cates = categorical_features)
    else:  # sklearn
        return sklearn_train(model, trainx, trainy, testx, predict_prob,eval_testx=eval_testx)

# k折train
def kfold_train(kfold, model_type, trainx, trainy, testx, params=None, use_valid=True, valid_ratio=0.2, validx=None,
                validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
                predict_prob=True, feature_importances=None, group=None, model_save_file=None,
                feature_names="auto", isFromFile=False, model=None, cate_threshold=10, use_all_data=False,
                all_data_model_weight=0.2, kfold_split_values=None, categorical_features='auto',
                oof_metric = None): # oof_metric is a dict, key is name, value is function(y_true, y_predict)
    if(kfold_split_values is None):
        kfold_split_values = trainy
    preds = []
    kf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
    eval_trainx = None
    if(oof_metric is not None):
        eval_res = {}
        for _k in oof_metric.keys():
            eval_res[_k] = []
    for _train, _test in kf.split(kfold_split_values, kfold_split_values):
        if(type(trainx) is np.ndarray):
            sub_trainx = trainx[_train]
            if(oof_metric is not None):
                eval_trainx = trainx[_test]
        else:
            sub_trainx = trainx.iloc[_train]
            if(oof_metric is not None):
                eval_trainx = trainx.iloc[_test]
        sub_trainy = trainy[_train]
        if(oof_metric is not None):
            eval_trainy = trainy[_test]
            pred, eval_predicts = general_train(model_type, sub_trainx, sub_trainy, testx,params, use_valid, valid_ratio, validx, validy,
                                 num_boost_round, early_stopping_rounds, random_state, predict_prob,
                                 feature_importances, group, model_save_file, feature_names, isFromFile,model,cate_threshold,
                                categorical_features=categorical_features, eval_testx= eval_trainx)
            for _k, _v in oof_metric.items():
                eval_res[_k].append(_v(eval_trainy, eval_predicts))

        else:
            pred = general_train(model_type, sub_trainx, sub_trainy, testx,params, use_valid, valid_ratio, validx, validy,
                                 num_boost_round, early_stopping_rounds, random_state, predict_prob,
                                 feature_importances, group, model_save_file, feature_names, isFromFile,model,cate_threshold,
                                 categorical_features=categorical_features, eval_testx= eval_trainx)
        preds.append(pred)
    pred = np.mean(preds, axis=0)

    if(use_all_data):
        _pred = general_train(model_type, trainx, trainy, testx,params, use_valid, valid_ratio, validx, validy,
                             num_boost_round, early_stopping_rounds, random_state, predict_prob,
                             feature_importances, group, model_save_file, feature_names, isFromFile,model,cate_threshold,
                             categorical_features=categorical_features)
        pred = (1-all_data_model_weight) * pred + all_data_model_weight * _pred
    if(oof_metric is None):
        return pred
    else:
        for _k in eval_res.keys():
            eval_res[_k] = np.mean(eval_res[_k])
        return pred, eval_res


# 半监督 train
def self_train(model_type, trainx, trainy, testx, params, use_valid=True, valid_ratio=0.2, validx=None,
               validy=None, num_boost_round=500, early_stopping_rounds=5, random_state=2018,
               predict_prob=True, feature_importances=None, group=None, model_save_file=None,
               feature_names="auto", isFromFile=False, model=None, cate_threshold=10, use_all_data=False, all_data_model_weight=0.2,
                min_prob=0.01,max_prob=0.99, max_iteration=1, kfold=1, rules=None):

    testx_idx = np.arange(testx.shape[0])
    adds = set()
    for _i in range(max_iteration):
        print("self train iteration:", _i)
        if(kfold > 1):
            print('use kfold train')
            pred = kfold_train(kfold, model_type,trainx,trainy,testx,params,use_valid,valid_ratio,
                               validx,validy,num_boost_round,early_stopping_rounds,random_state,
                               True,feature_importances,group,None,feature_names,isFromFile,model,cate_threshold,use_all_data,all_data_model_weight)
        else:
            print('use general train')
            pred = general_train(model_type,trainx,trainy,testx,params,use_valid,valid_ratio,
                                 validx,validy,num_boost_round,early_stopping_rounds,random_state,
                                 True, feature_importances, group,None,feature_names,isFromFile,model,cate_threshold)

        if(rules is not None):
            for _id, _v in rules.items():
                pred[_id] = _v
            rules = None

        add_temp1 = set(testx_idx[pred <= min_prob])
        add_temp2 = set(testx_idx[pred >= max_prob])
        add_temp = add_temp1.union(add_temp2)
        temp = list(add_temp - adds)
        adds = adds.union(add_temp)
        if(len(temp) > 0):
            print("add ", len(temp))
            idx = testx_idx[temp]
            trainx = np.concatenate([trainx, testx[idx]])
            trainy = np.concatenate([trainy, np.round(pred[idx]).astype(int)])
        else:
            print("no sample can be added")
            break
    print('final train')
    if(kfold > 1):
        print('use kfold train')
        pred = kfold_train(kfold, model_type,trainx,trainy,testx,params,use_valid,valid_ratio,
                           validx,validy,num_boost_round,early_stopping_rounds,random_state,
                           predict_prob,feature_importances,group,model_save_file,feature_names,isFromFile,model,cate_threshold,use_all_data,all_data_model_weight)
    else:
        print('use general train')
        pred = general_train(model_type,trainx,trainy,testx,params,use_valid,valid_ratio,
                             validx,validy,num_boost_round,early_stopping_rounds,random_state,
                             predict_prob, feature_importances, group,model_save_file,feature_names,isFromFile,model,cate_threshold)
    return pred