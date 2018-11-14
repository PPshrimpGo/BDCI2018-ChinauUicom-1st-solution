#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 11:01:29 2018

@author: hzs
"""

import pandas as pd
import numpy as np 

sub = pd.read_csv('./sub/sub_final.csv')


def replace_white_user_id(submission):
    white_user_id = pd.read_csv('white.csv')
    white_dict = dict(zip(white_user_id['user_id'],white_user_id['current_service'].astype(int)))
    submission['current_service'] = list(map(lambda x,y: y if x not in white_dict else white_dict[x],submission['user_id'],submission['current_service']))
    return submission

sub = replace_white_user_id(sub)
sub.to_csv('./sub_final_white.csv',index=None)


def f(x):
    try:
        r = x.split('.')[1][-1]
        return 0 if r=='0' else 1
    except:
        return 1
