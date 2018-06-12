#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-6-11, 22:19

@Description:

@Update Date: 18-6-11, 22:19
"""

from sklearn.model_selection import StratifiedKFold
from DMF.train_predict import lgb_train_model, lgb_train, sklearn_train, sklearn_train_model
import numpy as np


class StackingBaseModel(object):
    """
    训练stacking base model
    """

    def __init__(self, model, model_name, other_params, cv,
                 use_valid=True, valid_ratio=0.2,
                 shuffle=True, random_state=None):
        self.base_model = model
        self.base_model_name = model_name
        self.params = other_params
        self.cv = cv
        if (valid_ratio <= 0):
            self.use_valid = False
            self.valid_ratio = valid_ratio
        else:
            self.use_valid = use_valid
            self.valid_ratio = valid_ratio
        self.split = StratifiedKFold(n_splits=self.cv, shuffle=shuffle, random_state=random_state)
        self._clear()

    def _clear(self):
        self.newx = None
        self.paramss = []
        self.base_models = []
        if (self.base_model_name == "lgb"):
            for _i in range(self.cv):
                self.paramss.append(self.params.copy())

    def _reshape(self, nindex, newx):
        _d = list(zip(nindex, np.arange(nindex.shape[0])))
        _d = sorted(_d, key=lambda x: x[0])
        _d = [_i[1] for _i in _d]
        return newx[_d]

    def fit(self, trainx, trainy):
        self._clear()
        newxs = []
        indexs = []
        for _i, (_train_index, _test_index) in enumerate(self.split.split(trainx, trainy)):
            _trainx = trainx[_train_index]
            _trainy = trainy[_train_index]
            _testx = trainx[_test_index]
            indexs.append(_test_index)
            if (self.base_model_name == "lgb"):
                self.base_models.append(lgb_train_model(_trainx, _trainy,
                                                        self.paramss[_i],
                                                        use_valid=self.use_valid,
                                                        valid_ratio=self.valid_ratio))
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
        self.trainx_ = trainx
        return self

    def _predict(self, model, model_name, x):
        return model.predict(x)

    def _feature_importance(self, top_k_origin_feature):
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

    def transform(self, testx, top_k_origin_feature=10):
        if (self.newx_ is None):
            print("please fit first")
            exit(1)
        if(len(self.newx_.shape) > 1):
            nc = self.newx_.shape[1]
        else:
            nc = 1

        trainx = self.newx_.reshape(-1,nc)
        if(top_k_origin_feature > 0):
            _origin_index = self._feature_importance(top_k_origin_feature)
            trainx = np.concatenate([trainx, self.trainx_[:, _origin_index]], axis=1)
        testxs = []
        for model in self.base_models:
            testxs.append(self._predict(model, self.base_model_name, testx))
        _testx = np.mean(testxs, axis=0)
        _testx = _testx.reshape(-1,nc)
        if(top_k_origin_feature > 0):
            testx = np.concatenate([_testx,testx[:, _origin_index]], axis=1)
        else:
            testx = _testx
        return trainx, testx

    def fit_transfrom(self, trainx, trainy, testx, top_k_origin_feature=10):
        self.fit(trainx, trainy)
        return self.transform(testx, top_k_origin_feature)
