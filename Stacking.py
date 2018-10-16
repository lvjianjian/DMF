#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-6-11, 22:19

@Description:

@Update Date: 18-6-11, 22:19
"""

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from DMF.train_predict import *
import numpy as np
import pandas as pd

class StackingBaseModel(BaseEstimator):
    """
    训练stacking base model
    """

    def _init(self):
        self.newx = None
        self.paramss = []
        self.base_models = []
        if (self.base_model_name in ['xgb', 'lgb']):
            for _i in range(self.cv):
                self.paramss.append(self.other_params.copy())
        self.use_origin_all = False
        if (self.top_k_origin_feature < 0):
            self.use_origin_all = True

    def __init__(self, base_model, base_model_name, other_params, cv,
                 use_valid=True, valid_ratio=0.2,
                 shuffle=True, random_state=None, top_k_origin_feature=10,
                 predict_proba=True, cat_boost_threshold=10):
        self.top_k_origin_feature = top_k_origin_feature
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.other_params = other_params
        self.cv = cv
        self.shuffle = shuffle
        self.use_valid = use_valid
        self.valid_ratio = valid_ratio
        self.random_state = random_state
        if (self.valid_ratio <= 0):
            self.use_valid = False
        self.split = StratifiedKFold(n_splits=self.cv, shuffle=self.shuffle, random_state=self.random_state)
        self.set_predict_proba(predict_proba)
        self.cat_boost_threshold = cat_boost_threshold
        self.feature_name = None
        self.topk_feature_name = None

    def _reshape(self, nindex, newx):
        _d = list(zip(nindex, np.arange(nindex.shape[0])))
        _d = sorted(_d, key=lambda x: x[0])
        _d = [_i[1] for _i in _d]
        return newx[_d]

    def set_feature_names(self, feature_names):
        self.feature_name = feature_names

    def _sub(self, t, index):
        if(type(t) is np.ndarray):
            return t[index]
        else:
            return t.iloc[index]


    def fit(self, trainx, trainy):
        self._init()
        newxs = []
        indexs = []
        for _i, (_train_index, _test_index) in enumerate(self.split.split(trainx, trainy)):
            _trainx = self._sub(trainx, _train_index)
            _trainy = self._sub(trainy, _train_index)
            _testx = self._sub(trainx, _test_index)
            indexs.append(_test_index)
            if (self.base_model_name == "lgb"):
                self.base_models.append(lgb_train_model(_trainx, _trainy,
                                                        self.paramss[_i],
                                                        use_valid=self.use_valid,
                                                        valid_ratio=self.valid_ratio,
                                                        feature_names=self.feature_name))
                _newx = self._predict(self.base_models[_i], self.base_model_name, _testx)
            elif (self.base_model_name == "xgb"):
                self.base_models.append(xgb_train_model(_trainx, _trainy,self.paramss[_i],
                                                        use_valid=self.use_valid,
                                                        valid_ratio=self.valid_ratio,
                                                        feature_names=self.feature_name))
                _newx = self._predict(self.base_models[_i], self.base_model_name, _testx)
            elif (self.base_model_name == "catboost"):
                self.base_models.append(
                    catboost_train_model(self.base_model, _trainx, _trainy, self.cat_boost_threshold))
                _newx = self._predict(self.base_models[_i], self.base_model_name, _testx)
            else:
                self.base_models.append(sklearn_train_model(self.base_model, _trainx, _trainy))
                _newx = self._predict(self.base_models[_i], self.base_model_name, _testx)
            newxs.append(_newx)
        nindex = np.hstack(indexs)
        if (len(newxs[0].shape) == 1):
            newx = np.hstack(newxs)
        else:
            newx = np.vstack(newxs)
        newx = self._reshape(nindex, newx)
        self.newx_ = newx
        if(type(trainx) != np.ndarray):
            self.trainx_ = trainx.values
        else:
            self.trainx_ = trainx

        if (len(self.newx_.shape) > 1):
            nc = self.newx_.shape[1]
        else:
            nc = 1
        self.nc_ = nc
        trainx = self.newx_.reshape(-1, nc)
        if (self.use_origin_all):
            trainx = np.concatenate([trainx, self.trainx_], axis=1)
        else:
            self._origin_index = self._feature_importance(self.top_k_origin_feature)
            trainx = np.concatenate([trainx, self.trainx_[:, self._origin_index]], axis=1)
        self.trainx_ = trainx
        self.top_k_origin_feature = None
        return self

    def _predict(self, model, model_name, x):
        return predict(model, model_name, x, self.predict_proba)

    def set_predict_proba(self, predict_proba):
        self.predict_proba = predict_proba

    def _feature_importance(self, top_k_origin_feature):
        if (top_k_origin_feature == 0):
            return []
        f = None
        nf = None
        for _model in self.base_models:
            _f = _model.feature_importance()
            if (f is None):
                nf = _f.shape[0]
                f = _f
            else:
                f += _f
        _d = list(zip(f, range(nf)))
        _d = sorted(_d, key=lambda x: -x[0])
        _d = [_i[1] for _i in _d[:top_k_origin_feature]]
        return _d

    def transform(self, testx):
        if(type(testx) != np.ndarray):
            testx = testx.values
        else:
            testx = testx
        if (self.newx_ is None):
            print("please fit first")
            exit(1)
        testxs = []
        for model in self.base_models:
            testxs.append(self._predict(model, self.base_model_name, testx))
        _testx = np.mean(testxs, axis=0)
        _testx = _testx.reshape(-1, self.nc_)
        if (self.use_origin_all):
            testx = np.concatenate([_testx, testx], axis=1)
            self.topk_feature_name = self.base_models[0].feature_name()
        else:
            testx = np.concatenate([_testx, testx[:, self._origin_index]], axis=1)
            if(self.base_model_name != 'sklearn'):
                if(self.base_model_name == 'lgb'):
                    fn = self.base_models[0].feature_name()
                    self.topk_feature_name = [fn[_i] for _i in self._origin_index]
                elif(self.base_model_name == 'xgb'):
                    fn = self.base_models[0].feature_names
                    self.topk_feature_name = [fn[_i] for _i in self._origin_index]
        return testx

    def fit_transfrom(self, trainx, trainy, testx):
        self.fit(trainx, trainy)
        return self.transform(testx)


class Stacking(BaseEstimator):
    def __init__(self, base_models, second_model, second_model_name, second_other_params,
                 use_valid=True, valid_ratio=0.2, random_state=None):
        self.base_models = base_models
        self.second_model = second_model
        self.second_model_name = second_model_name
        self.second_other_params = second_other_params
        self.use_valid = use_valid
        self.valid_ratio = valid_ratio
        self.random_state = random_state

    def fit(self, trainx, trainy):
        trainxs = []
        for _basemodel in self.base_models:
            _basemodel.fit(trainx, trainy)
            trainxs.append(_basemodel.trainx_)
        trainx = np.concatenate(trainxs, axis=1)
        if (self.second_model_name == "lgb"):
            self.second_model = lgb_train_model(trainx, trainy, self.second_other_params,
                                                use_valid=self.use_valid, valid_ratio=self.valid_ratio,
                                                random_state=self.random_state)

    def predict(self, testx):
        testxs = []
        for _basemodel in self.base_models:
            testxs.append(_basemodel.transform(testx))
        testx = np.concatenate(testxs, axis=1)
        return self.second_model.predict(testx)
