#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-7-7, 13:39

@Description:

@Update Date: 18-7-7, 13:39
"""

from data import *
from DMF.train_predict import *
from DMF.util import concat
from sklearn import metrics
from feature import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, roc_auc_score
import zipfile

def save(res, dataPath, save_file_name):
    resPath = os.path.join(dataPath, "result")
    if (not os.path.exists(resPath)):
        os.mkdir(resPath)
    path = os.path.join(resPath, save_file_name)
    res['id'] = res['id'].astype(int)
    res[['id', 'probability']].to_csv(path, index=None)

def reduce_file_name(name):
    # 缩小文件名
    names = name.split("_")
    fname = ""
    for _n in names:
        fname += (_n[:4] + "_")
    return fname[:-1]

@performance
def run(valid):
    dataPath = '../data/'
    useValid = False
    temp = labels(dataPath)
    nsample = temp.shape[0]
    feature_functions = [
        f_type_feats,
        dist_basic_feats,
    ]

    feature_names = []
    save_name = ""
    features = []
    for _fun in feature_functions:
        save_name += reduce_file_name(_fun.__name__.replace("_feature", "")) + "_"
        if (_fun in NEED_VALID_FUN):
            temp2 = _fun(valid, dataPath)
        else:
            temp2 = _fun(dataPath)
        assert (temp2.shape[0] == nsample)
        features.append(temp2)
    features2 = concat(features)
    
    del temp2
    del features
    features = features2
    del features2
    feature_names += list(features.columns)
    features[temp.columns.tolist()] = temp
    
    del temp; gc.collect()
    temp = features
    print(temp.shape)
    del features
    gc.collect()
    
    save_name += "{}.csv"
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 32,
        'learning_rate': 0.5,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1,
        'seed':2019
    }
    
    num_boost_round = 1000
    
    _cate_ids = []
    
    feature_names = [_f for _f in feature_names if _f not in ['_index', 'type', 'scalar_coupling_constant', 'id', 'original_type']]
    
    cate_ids = []
    for _i,_c in enumerate(_cate_ids):
        if(_c in feature_names):
            cate_ids.append(_c)
    cate_id_indexs=[]
    for _i,_c in enumerate(feature_names):
        if(_c in cate_ids):
            cate_id_indexs.append(_i)
#     if(len(cate_ids) == 0):
#         cate_ids = 'auto'
        
    print("start train")
    
    if(valid):
        temp = temp[temp.type!='test']
        temp.loc[temp.type=='valid','type'] = 'test'
        print(temp['type'].unique())
    
    
    trainx = temp[(temp.type == "train") | (temp.type == "valid")]
    testx = temp[temp.type == 'test']
        
    trainy = trainx["scalar_coupling_constant"].values
    testy = testx['scalar_coupling_constant'].values
    
    del temp; gc.collect()
    if(not valid):
        record_file = open(time.strftime("%Y-%m-%d_feature_importance_online") + '.txt', 'a+')
    else:
        record_file = open(time.strftime("%Y-%m-%d_feature_importance_offline") + '.txt', 'a+')
        
    feature_importances = []
    print(len(feature_names))
        
    if(valid):
        res = general_train('lgb',trainx[feature_names], trainy, testx[feature_names], params, False,
                            validx = testx[feature_names],validy = testy,
                        num_boost_round=num_boost_round, isFromFile=False, categorical_features=cate_id_indexs,
                        early_stopping_rounds=50,feature_names=feature_names,feature_importances=feature_importances)
    else:
        res = general_train('lgb',trainx[feature_names], trainy, testx[feature_names], params, False, num_boost_round=num_boost_round, isFromFile=False, categorical_features=cate_ids,feature_importances=feature_importances)
    
    
    for _f in feature_importances:
        print(_f)
        print(_f, file=record_file)
    
    
    testx['scalar_coupling_constant_predict'] = res
    
    if(valid):
        maes = []
        for _type in testx['original_type'].unique():
            _mae = np.mean(np.abs(testx[testx['original_type'] == _type]['scalar_coupling_constant_predict'].values - testx[testx['original_type'] == _type]['scalar_coupling_constant'].values))
            maes.append(_mae)
            print(_type, _mae)
        mae = np.mean(maes)
        print('mae', mae)
        print('mae', mae, file = record_file)
    else:
        save_name = 'submission_{}r_neg_state_{}.csv'.format(num_boost_round,negetive_user_sample_random_sate)
        print('save in ', save_name)
        save(testx, dataPath, save_name)


if __name__ == '__main__':
    run(True)
