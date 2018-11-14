#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 23:08:26 2018

@author: bmj
"""

import gc
import time
from time import strftime,gmtime
import numpy as np
import pandas as pd
import os
load = False
cache_path = './cache3/'

from time import strftime,gmtime


def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            try:
                result[l.columns.tolist()] = l
            except:
                print(l.head())
    return result


def left_merge(data1,data2,on):
    if type(on) != list:
        on = [on]
    if (set(on) & set(data2.columns)) != set(on):
        data2_temp = data2.reset_index()
    else:
        data2_temp = data2.copy()
    columns = [f for f in data2.columns if f not in on]
    result = data1.merge(data2_temp,on=on,how='left')
    result = result[columns]
    return result


def get_feat_size(train,size_feat):
    """计算A组的数量大小（忽略NaN等价于count）"""
    result_path = cache_path +  ('_').join(size_feat)+'_feat_count'+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        result = train[size_feat].groupby(by=size_feat).size().reset_index().rename(columns={0: ('_').join(size_feat)+'_count'})
        result = left_merge(train,result,on=size_feat)
    return result


def get_feat_size_feat(train,base_feat,other_feat):
    """计算唯一计数（等价于unique count）"""
    result_path = cache_path + ('_').join(base_feat)+'_count_'+('_').join(other_feat)+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        result = train[base_feat].groupby(base_feat).size().reset_index()\
                      .groupby(other_feat).size().reset_index().rename(columns={0: ('_').join(base_feat)+'_count_'+('_').join(other_feat)})
        result = left_merge(train,result,on=other_feat)
    return result


def get_feat_stat_feat(train,base_feat,other_feat,stat_list=['min','max','var','size','mean','skew']):
    name = ('_').join(base_feat) + '_' + ('_').join(other_feat) + '_' + ('_').join(stat_list)
    result_path = cache_path + name +'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path)
    else:
        agg_dict = {}
        for stat in stat_list:
            agg_dict[name+stat] = stat
        result = train[base_feat + other_feat].groupby(base_feat)[",".join(other_feat)]\
        .agg(agg_dict)
        result = left_merge(train,result,on=base_feat)
    return result

def get_feat_ngroup(train,base_feat):
    name = ('_').join(base_feat)+'_ngroup'
    result_path = cache_path + ('_').join(base_feat)+'_ngroup'+'.hdf'
    if os.path.exists(result_path) & load:
        result = pd.read_hdf(result_path, 'w')
    else:
        train[name] = train.groupby(base_feat).ngroup()
        result = train[[name]]
        train.drop([name],axis=1,inplace=True)        
    return result
