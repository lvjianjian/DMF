#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 18-5-12, 10:40

@Description:

@Update Date: 18-5-12, 10:40
"""
import pandas as pd
import os
import numpy as np
from DMF.util import performance, downcast, mkpath
import pickle
from multiprocessing import Process
import multiprocessing
from collections import Counter
import scipy
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AffinityPropagation, KMeans, MiniBatchKMeans
from sklearn.decomposition import NMF, PCA, TruncatedSVD
import gc
import time
import pickle
from DMF.smooth import HyperParam
from joblib import *
import json
from tqdm import tqdm
from DMF.util import dump_feature_remove_main_id, euclidean, MAX_DIS_FUNCTION, cosine,concat
import datetime

@performance
def all_data(dataPath):
    path = os.path.join(dataPath,'data.pickle')
    if(not os.path.exists(path)):
        structures = structures_data(dataPath)
        train = pd.read_csv(f'{dataPath}/train.csv')
        test = pd.read_csv(f'{dataPath}/test.csv')
        test['scalar_coupling_constant'] = -1
        train['f_type'] = train['type']
        test['f_type'] = test['type']
        train['type'] = 'train'
        _valid = pd.DataFrame(train['molecule_name'].unique()).sample(frac=0.2,random_state=2019)[0]
        train.loc[train[train.molecule_name.isin(_valid)].index,'type'] = 'valid'
        test['type'] = 'test'
        data = pd.concat([train, test]).reset_index(drop=True)
        print(data['type'].value_counts())
        data['_index'] = data.index 
        data = map_atom_info(data, 0, structures)
        data = map_atom_info(data, 1, structures)
        data.to_pickle(path)
    else:
        data = pd.read_pickle(path)
    return data

def map_atom_info(df, atom_idx, structures):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


@performance
def structures_data(dataPath):
    return pd.read_csv(f'{dataPath}/structures.csv')

if __name__ == '__main__':
    all_data('../data/')
    structures_data('../data/')
