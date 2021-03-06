import numpy as np
import lightgbm as lgb
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import warnings
from joblib import Parallel, delayed
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import math
import gc
import random
import operator
import matplotlib
import math
import collections
import scipy.special as special

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
import itertools
import numpy


class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def samples_from_beta(self, alpha, beta, num, imp_upperbound):
        samples = numpy.random.beta(alpha, beta, num)
        imps = []
        clicks = []
        for click_ratio in samples:
            _imp = imp_upperbound * random.random()
            _click = click_ratio * _imp
            imps.append(_imp)
            clicks.append(_click)
        return imps, clicks

    def update_from_data_by_FPI(self, all_num, click_num, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(all_num, click_num, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, all_num, click_num, alpha, beta):
        '''fixed point iteration'''
        sum_fenzi_alpha = (special.digamma(click_num + alpha) - special.digamma(alpha)).sum()
        sum_fenzi_beta = ((special.digamma(all_num - click_num + beta) - special.digamma(beta))).sum()
        sum_fenmu = ((special.digamma(all_num + alpha + beta) - special.digamma(alpha + beta))).sum()
        return alpha * (sum_fenzi_alpha / sum_fenmu), beta * (sum_fenzi_beta / sum_fenmu)

    def update_from_data(self, all_num, click_num):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute(all_num, click_num)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def  __compute(self, all_num, click_num):
        '''moment estimation'''
        mean = (click_num / all_num).mean()
        if len(all_num) == 1:
            var = 0
        else:
            var = (click_num / all_num).var()
        return mean, var
