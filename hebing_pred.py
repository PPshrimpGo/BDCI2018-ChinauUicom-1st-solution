#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:49:27 2018

@author: hzs
"""
import pandas as pd

train_p4 =  pd.read_csv('./cv/val_prob_model_3_1.csv')
test_p4 =  pd.read_csv('./cv/sub_prob_model_3_1.csv')

train_p1 =  pd.read_csv('./cv/val_prob_model_3_4.csv')
test_p1 =  pd.read_csv('./cv/sub_prob_model_3_4.csv')

train_p = pd.concat([train_p4,train_p1])
test_p = pd.concat([test_p4,test_p1])

train_p = train_p.to_csv('val_prob_hebing2.csv',index=None)
test_p = test_p.to_csv('sub_prob_hebing2.csv',index=None)