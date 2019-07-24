#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-7-7, 13:39

@Description:

@Update Date: 18-7-7, 13:39
"""

from data import *
import numpy as np
import pandas as pd
from DMF.util import dump_feature_remove_main_id, euclidean, MAX_DIS_FUNCTION, cosine,concat
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from itertools import product
import pickle
import gc
import time
import math
from multiprocessing import Process
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from DMF.Stacking import StackingBaseModel
from DMF.util import *
from DMF.smooth import HyperParam
from sklearn.decomposition import PCA as pca
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD,NMF
import scipy
from DMF import cate_embedding
import json
from tqdm import tqdm
from DMF.MeanEncoder import *

NEED_VALID_FUN = set()

@dump_feature_remove_main_id
def labels(dataPath):
    data = all_data(dataPath)
    data['original_type'] = data['f_type']
    return data[['_index', 'type', 'scalar_coupling_constant', 'id', 'original_type']]

@dump_feature_remove_main_id
def f_type_feats(dataPath):
    data = all_data(dataPath)
    data['f_type_0'] = data['f_type'].apply(lambda x: x[0])
    data['f_type_1'] = data['f_type'].apply(lambda x: x[2:])
    for f in ['atom_0', 'atom_1', 'f_type_0', 'f_type_1', 'f_type']:
        lbl = LabelEncoder()
        lbl.fit(data[f].values)
        data[f] = lbl.transform(list(data[f].values))
    return data[['_index', 'f_type', 'f_type_0', 'f_type_1', 'atom_0', 'atom_1', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1']]

@dump_feature_remove_main_id
def dist_basic_feats(dataPath):
    data = all_data(dataPath)

    data_p_0 = data[['x_0', 'y_0', 'z_0']].values
    data_p_1 = data[['x_1', 'y_1', 'z_1']].values

    data['dist_xyz'] = np.linalg.norm(data[['x_0', 'y_0', 'z_0']].values - data[['x_1', 'y_1', 'z_1']].values, axis=1)
    data['dist_xy'] = np.linalg.norm(data[['x_0', 'y_0']].values - data[['x_1', 'y_1']].values, axis=1)
    data['dist_xz'] = np.linalg.norm(data[['x_0', 'z_0']].values - data[['x_1', 'z_1']].values, axis=1)
    data['dist_yz'] = np.linalg.norm(data[['y_0', 'z_0']].values - data[['y_1', 'z_1']].values, axis=1)
    data['dist_x'] = np.linalg.norm(data[['x_0']].values - data[['x_1']].values, axis=1) 
    data['dist_y'] = np.linalg.norm(data[['y_0']].values - data[['y_1']].values, axis=1)
    data['dist_z'] = np.linalg.norm(data[['z_0']].values - data[['z_1']].values, axis=1) 
    
    return data[['_index','dist_xyz','dist_x','dist_y','dist_z','dist_xy','dist_xz','dist_yz']]

@dump_feature_remove_main_id
def structure_basic_feats(dataPath):
    structures = structures_data(dataPath)
    temp = structures.groupby(['molecule_name','atom'])['atom'].agg({'c':'count'})
    temp = temp.pivot_table(index='molecule_name',columns='atom').reset_index().fillna(0)
    res = temp['c']
    res['molecule_name'] = temp['molecule_name']
    temp2 = structures.groupby('molecule_name')['atom'].agg({'atom_num':'count','atom_nunique':'nunique'})
    res = res.merge(temp2,on = 'molecule_name', how='left')
    fns = ['atom_num','atom_nunique','molecule_name']
    for _t in ['C', 'F', 'H', 'N', 'O']:
        res[f'{_t}_num'] = res[_t].astype(int)
        res[f'{_t}_ratio'] = res[f'{_t}_num'] / res['atom_num']
        fns.append(f'{_t}_num')
        fns.append(f'{_t}_ratio')
    res = res[fns]
    data = all_data(dataPath)[['_index','molecule_name']]    
    data = data.merge(res, on ='molecule_name', how ='left')
    del data['molecule_name']
    return data

# @dump_feature_remove_main_id
# def structure_basic_feats2(dataPath):
#     feats = structure_basic_feats(dataPath)
#     feats['HFO_IV'] = feats['H_num'] * 1 + feats['F_num'] * -1 + feats['O_num'] * -2
    
#     return feats

@dump_feature_remove_main_id
def single_feats1(dataPath):
    fns = []
    data = all_data(dataPath)
    f = dist_basic_feats(dataPath)
    data[f.columns.tolist()] = f
    f = f_type_feats(dataPath)[['f_type', 'f_type_0', 'f_type_1']]
    data[f.columns.tolist()] = f
    cat_cols = ['f_type', 'f_type_0', 'f_type_1']
    for col in cat_cols:
        data[f'molecule_{col}_nunique'] = data.groupby('molecule_name')[col].transform('nunique')
        fns.append(f'molecule_{col}_nunique')
        
    num_cols = ['x_0','y_0','z_0','x_1', 'y_1', 'z_1', 'dist_xyz', 'dist_x', 'dist_y', 'dist_z']    
    aggs = ['mean', 'max', 'std', 'min', 'median']    
    for col in num_cols:
        for agg in aggs:
            data[f'molecule_{col}_{agg}'] = data.groupby('molecule_name')[col].transform(agg)
            fns.append(f'molecule_{col}_{agg}')
            
    fns.append('_index')
    return data[fns]

@dump_feature_remove_main_id
def double_feats1(dataPath):
    fns = []
    data = all_data(dataPath)
    f = dist_basic_feats(dataPath)
    data[f.columns.tolist()] = f
    f = f_type_feats(dataPath)[['f_type', 'f_type_0', 'f_type_1']]
    data[f.columns.tolist()] = f
    
    num_cols = ['x_0','y_0','z_0','x_1', 'y_1', 'z_1', 'dist_xyz', 'dist_x', 'dist_y', 'dist_z']
    cat_cols = ['f_type', 'atom_1','atom_0']
    aggs = ['mean', 'max', 'std', 'min', 'median']
    
    for cat_col in tqdm(cat_cols):
        for num_col in num_cols:
            for agg in aggs:
                data[f'molecule_{cat_col}_{num_col}_{agg}'] = data.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                data[f'molecule_{cat_col}_{num_col}_{agg}_diff'] = data[f'molecule_{cat_col}_{num_col}_{agg}'] - data[num_col]
                data[f'molecule_{cat_col}_{num_col}_{agg}_div'] = data[f'molecule_{cat_col}_{num_col}_{agg}'] / data[num_col]
                fns.append(f'molecule_{cat_col}_{num_col}_{agg}')
                fns.append(f'molecule_{cat_col}_{num_col}_{agg}_diff')
                fns.append(f'molecule_{cat_col}_{num_col}_{agg}_div')
    fns.append('_index')
    return data[fns]
            
def create_features_full(df):
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_1'])['id'].transform('count')

    num_cols = ['x_0','y_0','z_0','x_1', 'y_1', 'z_1', 'dist', 'dist_x', 'dist_y', 'dist_z']
    cat_cols = ['atom_index_0', 'atom_index_1', 'type', 'atom_1', 'type_0', 'type_1']
    aggs = ['mean', 'max', 'std', 'min', 'median']
    for col in cat_cols:
        df[f'molecule_{col}_nunique'] = df.groupby('molecule_name')[col].transform('nunique')
        
    for col in num_cols:
        for agg in aggs:
            df[f'molecule_{col}_{agg}'] = df.groupby('molecule_name')[col].transform(agg)
    
    for cat_col in tqdm(cat_cols):
        for num_col in num_cols:
            for agg in aggs:
                df[f'molecule_{cat_col}_{num_col}_{agg}'] = df.groupby(['molecule_name', cat_col])[num_col].transform(agg)
                df[f'molecule_{cat_col}_{num_col}_{agg}_diff'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] - df[num_col]
                df[f'molecule_{cat_col}_{num_col}_{agg}_div'] = df[f'molecule_{cat_col}_{num_col}_{agg}'] / df[num_col]
                
    df = reduce_mem_usage(df)
    return df

if __name__ == '__main__':
    labels('../data/')
