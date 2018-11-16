#import dask.dataframe as dd
#from dask.multiprocessing import get
import itertools
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error,mean_absolute_error, f1_score
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
#import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from utils import *
##from utils3 import *
from datetime import datetime
from datetime import timedelta
#from tqdm import tqdm
#test

warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

USE_KFOLD = True

def process_data(data,is_train=False):
    
    def parse_genser(x):
        if x == '01':
            return '0'
        elif x == '02':
            return '1'
            
        elif x == '00':
            return np.nan
        else:
            return x
    
    print(data.gender.value_counts())
    data['gender'] = data['gender'].apply(lambda x: parse_genser(x))
    print(data.gender.value_counts())  
    
    for col in ['gender', 'age']:
        data[col] = data[col].replace("\\N",np.nan)
        data[col] = data[col].astype('float')

    
    if is_train:
        pass

    return data
    

train = pd.read_csv('./input/train.csv',dtype = {'gender':str})[:]
train = train[train.current_service!=999999]
#train = train.iloc[:520870]

train = process_data(train,is_train=True)
print(len(train))

test = pd.read_csv('./input/test.csv',dtype = {
    'gender':str,
    '1_total_fee':str,
    '2_total_fee':str,
    '3_total_fee':str,
    '4_total_fee':str,
     'month_traffic':str, 
     'last_month_traffic':str, 
     'local_trafffic_month':str, 
    'local_caller_time':str, 
    'service1_caller_time':str, 
    'service2_caller_time':str,
    'pay_num':str,}
                     )[:]
test = process_data(test,is_train=False)
test.drop(['1_total_fee'] ,axis = 1, inplace = True)

train_old = pd.read_csv('./input/train_old.csv',dtype = {
    'gender':str,
    '1_total_fee':str,
    '2_total_fee':str,
    '3_total_fee':str,
    '4_total_fee':str,
    'month_traffic':str, 
     'last_month_traffic':str, 
     'local_trafffic_month':str, 
    'local_caller_time':str, 
    'service1_caller_time':str, 
    'service2_caller_time':str,
    'pay_num':str,}
                     )[:]
train_old = process_data(train_old,is_train=False)
train = train.append(train_old).reset_index(drop = True)
print(len(train))
shape1 = len(train)
train = train.append(test).reset_index(drop = True)
print(len(train))
shape2 = len(train)


#train = train[train['service_type']!=1]

import_continue = [
    '1_total_fee',
    '3_total_fee',
    'last_month_traffic',
    'service2_caller_time',
    'fea-sum',
    '2month_traffic_sum',
    'contract_time',
    'fea-min'
    ]


call_time = ['local_caller_time', 'service1_caller_time', 'service2_caller_time']
traffic = ['month_traffic','last_month_traffic','local_trafffic_month']
cat_cols = [
    'service_type',          #2,3
    'is_mix_service',        #2
    'many_over_bill',        #2
    'contract_type',         #9,8
    'is_promise_low_consume',#2
    'net_service',           #4
    'gender',                #3
    'complaint_level',       #4
        
    'cut_online',
    'cut_age',
    
    ]
continus_col = [
    '1_total_fee', '2_total_fee', '3_total_fee',  '4_total_fee', 'pay_num','former_complaint_fee',
    
    'month_traffic', 'last_month_traffic', 'local_trafffic_month', 
    
    'local_caller_time', 'service1_caller_time', 'service2_caller_time',
    
    'online_time','contract_time',  
     
    'pay_times', 'former_complaint_num'
    ]
def one_hot_encoder(train,column,n=100,nan_as_category=False):
    tmp = train[column].value_counts().to_frame()
    values = list(tmp[tmp[column]>n].index)
    train.loc[train[column].isin(values),column+'N'] = train.loc[train[column].isin(values),column]
    train =  pd.get_dummies(train, columns=[column+'N'], dummy_na=False)
    return train

def get_len_after_decimal(x):
    try:
        r = len(x.split('.')[1])
    except:
        r = np.nan
    return r

def get_len_before_decimal(x):
    try:
        r = len(x.split('.')[0])
    except:
        r = np.nan
    return r

def get_len_00(x):
    try:
        r = 1 if '.00' == x[-3:] else 0
    except:
        r = np.nan
    return r

def get_len_0(x):
    try:
        r = 1 if '.0' == x[-3:] else 0
    except:
        r = np.nan
    return r

def pasre_fee(x):
    taocan = [16,26,36,46,56,66,76,106,136,166,196,296,396,596]
    for i in range(len(taocan)):
        if x - taocan[i] >= 15:
            return i
    return np.nan

def pasre_fee_min(x):
    taocan = [16,26,36,46,56,66,76,106,136,166,196,296,396,596]
    for i in range(len(taocan)):
        if abs(x - taocan[i]) <  0.01:
            return i
    return np.nan

result = []


for column in ['2_total_fee', '3_total_fee',  '4_total_fee']:
    print(column, train[column].dtypes)
    #train[column+'_len_after_dot'] = train[column].apply(lambda x: get_len_after_decimal(x))
    train[column+'_len_before_dot'] = train[column].apply(lambda x: get_len_before_decimal(x))
    train[column+'_00'] = train[column].apply(lambda x: get_len_00(x))
    train[column+'_0'] = train[column].apply(lambda x: get_len_0(x))
    train[column] = train[column].replace("\\N",np.nan)
    train[column] =  train[column].astype('float')


#
#for column in continus_col:
#    train = one_hot_encoder(train,column,n=1000,nan_as_category=True)
train['fea-min'] = train[[str(1+i) +'_total_fee' for i in range(2,4)]].min(axis = 1)

for column in ['2_total_fee', '3_total_fee',  '4_total_fee', 'pay_num','fea-min']:
    train[column] =  train[column].astype('float')
    train[column+'_parse_min'] = train[column].apply(lambda x: pasre_fee_min(x))
    train[column+'_parse'] = train[column].apply(lambda x: pasre_fee(x))
    train[column+'_int'] = round(train[column].fillna(0)).astype('int')
    train[column+'_shifenwei'] = train[column+'_int'] // 10  
    train[column+'_int_last'] = train[column+'_int']%10 #last int 
    train[column+'_decimal'] = ((train[column+'_int'] - train[column])*100).fillna(0).astype('int')    #decimal
    train[column+'_decimal_is_0'] = (train[column+'_decimal']==0).astype('int')
    train[column+'_decimal_is_5'] = (train[column+'_decimal']%5==0).astype('int') 
    train[column+'_decimal_last'] = train[column+'_decimal']%10    
###    train = one_hot_encoder(train,column,n=2000,nan_as_category=True)
train['pay_num_last2'] = train['pay_num_int']%100  
train['former_complaint_fee_last2'] = round(train['former_complaint_fee'])%100






for column in ['month_traffic', 'last_month_traffic', 'local_trafffic_month']:
    #train[column+'_len_after_dot'] = train[column].apply(lambda x: get_len_after_decimal(x))
    train[column+'_len_before_dot'] = train[column].apply(lambda x: get_len_before_decimal(x))    
    train[column] =  round(train[column].astype('float'),6)
    train[column+'_is_int'] = ((train[column]%1 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_512'] = ((train[column]%512 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_50'] = ((train[column]%50 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_double'] = ((train[column]%512%50 == 0)&(train[column] != 0)).astype('int')

for column in ['local_caller_time', 'service1_caller_time', 'service2_caller_time','service12']:
    if column == 'service12':
        train['service12'] = train['service2_caller_time']+train['service1_caller_time']
    else:
        #train[column+'_len_after_dot'] = train[column].apply(lambda x: get_len_after_decimal(x))
        train[column+'_len_before_dot'] = train[column].apply(lambda x: get_len_before_decimal(x))
    train[column] =  round(train[column].astype('float'),6)
    train[column+'_decimal'] =  round(((round(train[column])- train[column])*60)).astype('int')
    train[column+'_decimal_is_int'] = ((train[column+'_decimal']==0)&(train[column] != 0)).astype('int')


print(train.shape)

train['is_duplicated'] = train.duplicated(subset=['service_type', 'is_mix_service', 'online_time', '1_total_fee',
       '2_total_fee', '3_total_fee', '4_total_fee','many_over_bill', 'contract_type', 'contract_time',
       'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num','local_caller_time',
       'service1_caller_time', 'service2_caller_time', 'gender', 'age'],keep=False)

#单特征处理
#年龄
train['age'] = train['age'].fillna(-20)
train['cut_age'] = train['age'].apply(lambda x: int(x/10))
train['cut_online'] = (train['online_time'] / 12).astype(int)


#同类特征加减乘除
#钱
train['4-fea-dealta'] = train['4_total_fee'] - train['3_total_fee']
train['3-fea-dealta'] = train['3_total_fee'] - train['2_total_fee']

train['4-fea-dealta_'] = train['4_total_fee'] / (train['3_total_fee']+0.00001)
train['3-fea-dealta_'] = train['3_total_fee'] / (train['2_total_fee']+0.00001)


#流量
train['month_traffic_delata'] = train['month_traffic'] - train['last_month_traffic']
train['month_traffic_delata_'] = train['month_traffic'] / (train['last_month_traffic']+0.00001)
train['2month_traffic_sum'] = train['month_traffic'] + train['last_month_traffic']
train['add_month_traffic'] = train['month_traffic'] - train['local_trafffic_month']
train['add_month_traffic'] = train['month_traffic'] / (train['local_trafffic_month']+0.00001)

#通话时间
train['service1_caller_time_delata'] = train['service1_caller_time'] / (train['service2_caller_time']+0.00001)
train['service1_caller_time_delata2'] = train['service1_caller_time'] / (train['local_caller_time']+0.00001)
train['service2_caller_time_delata_'] = train['service2_caller_time'] / (train['local_caller_time']+0.00001)

#合约时间
train['div_online_time_contract'] = train['contract_time'] / (train['online_time']+0.00001)
train['div_online_time_contract'] = train['contract_time'] - train['online_time']

#次数
train['div_former_complaint_num'] = train['former_complaint_num'] / (train['pay_times']+0.00001)
train['div_former_complaint_num'] = train['former_complaint_num'] - train['pay_times']

#同类特征统计
#4个月的费用和
train['fea-sum'] = train[[str(1+i) +'_total_fee' for i in range(2,4)]].sum(axis = 1)
train['fea-var'] = train[[str(1+i) +'_total_fee' for i in range(2,4)]].var(axis = 1)
train['fea-max'] = train[[str(1+i) +'_total_fee' for i in range(2,4)]].max(axis = 1)
train['fea-min'] = train[[str(1+i) +'_total_fee' for i in range(2,4)]].min(axis = 1)

train['call_time_sum'] = train[call_time].sum(axis = 1)
train['call_time_var'] = train[call_time].var(axis = 1)
train['call_time_min'] = train[call_time].min(axis = 1)
train['call_time_max'] = train[call_time].max(axis = 1)

train['traffic_sum'] = train[traffic].sum(axis = 1)
train['traffic_var'] = train[traffic].var(axis = 1)
train['traffic_min'] = train[traffic].min(axis = 1)
train['traffic_max'] = train[traffic].max(axis = 1)

#不同类特征加减乘除
#钱_次数
train['average_pay'] = train['pay_num'] / train['pay_times']

#钱_流量
train['div_traffic_price_2'] = train['last_month_traffic']/ 1000 / train['2_total_fee']

#钱_通话时间


#流量_通话时间
train['div_service1_caller_time'] = train['service1_caller_time']/train['last_month_traffic']
train['div_local_caller_time'] = train['local_caller_time']/train['last_month_traffic']
train['div_local_caller_time2'] = train['local_caller_time']/train['month_traffic']

#费用_次数
train['avg_complain_fee'] = train['former_complaint_fee'] / (train['former_complaint_num'] + 0.000000001)

# cat*num

result.append(get_feat_ngroup(train,['cut_age','gender']))
#result.append(get_feat_stat_feat(train, ['contract_type'], ['1_total_fee'], ['max']))
result.append(get_feat_stat_feat(train, ['contract_type'], ['2_total_fee'], ['mean']))
result.append(get_feat_stat_feat(train, ['contract_type'], ['last_month_traffic'], ['var','mean']))
result.append(get_feat_stat_feat(train, ['contract_type'], ['call_time_sum'], ['mean']))

for base_feat in [['contract_type']]:
    for other_feat in ['pay_num',
                         'month_traffic', 'last_month_traffic', 'local_trafffic_month', 
                         'local_caller_time', 'service1_caller_time', 'service2_caller_time',
                       ]:
        stat_list = ['mean']
        tmp = get_feat_stat_feat(train,base_feat,[other_feat],stat_list=stat_list)
        name = tmp.columns[0]
        train[name] = tmp
        train[name+'_comp'] = train[other_feat].values-train[name].values

#比例性特征
#train['1_total_fee_ratio'] = train['1_total_fee']/(train['fea-sum']+0.000001)
train['3_total_fee_ratio'] = train['3_total_fee']/(train['fea-sum']+0.000001)
train['call_time_sum_ratio'] = train['call_time_sum']/(train['traffic_sum']+0.000001) 
train['call_time_sum_ratio2'] = train['call_time_sum']/(train['fea-sum']+0.000001) 
train['traffic_sum_ratio1'] = train['traffic_sum']/(train['fea-sum']+0.000001) 



def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1),axis=0)
    score_vali = f1_score(y_true=labels,y_pred=preds,average='macro')
    return 'macro_f1_score', score_vali, True

def evaluate_macroF1_lgb(data_vali, preds):  
    labels = data_vali.astype(int)
    preds = np.array(preds)
    preds = np.argmax(preds.reshape(11, -1),axis=0)
    score_vali = f1_score(y_true=labels,y_pred=preds,average='macro')
    return  score_vali

def kfold_lightgbm(params,df, predictors,target,num_folds, stratified = False,
                   objective='', metrics='',debug= False,
                   feval = None, early_stopping_rounds=120, num_boost_round=100, verbose_eval=50, categorical_features=None,sklearn_mertric = evaluate_macroF1_lgb ):

    lgb_params = params
    
    train_df = df[df[target].notnull()]
    test_df = df[df[target].isnull()]
    
    # Divide in training/validation and test data
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df[predictors].shape, test_df[predictors].shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1234)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1234)
#    folds = GroupKFold(n_splits=5)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = predictors
    cv_resul = []
    '''
    perm = [i for i in range(len(train_df))]
    perm = pd.DataFrame(perm)
    perm.columns = ['index_']

    for n_fold in range(5):
        train_idx = np.array(perm[train_df['cv'] != n_fold]['index_'])
        valid_idx = np.array(perm[train_df['cv'] == n_fold]['index_'])
    '''
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df[target])):
        if (USE_KFOLD == False) and (n_fold == 1):
            break
        train_x, train_y = train_df[feats].iloc[train_idx], train_df[target].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df[target].iloc[valid_idx]
        
        #train_x = pd.concat([train_x,train_old[feats]])
        #train_y = pd.concat([train_y,train_old[target]])

        train_y_t = train_y.values
        valid_y_t = valid_y.values
        print(train_y_t)
        xgtrain = lgb.Dataset(train_x.values, label = train_y_t,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )
        xgvalid = lgb.Dataset(valid_x.values, label = valid_y_t,
                              feature_name=predictors,
                              categorical_feature=categorical_features
                              )

        clf = lgb.train(lgb_params, 
                         xgtrain, 
                         valid_sets=[xgvalid],#, xgtrain], 
                         valid_names=['valid'],#,'train'], 
                         num_boost_round=num_boost_round,
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=verbose_eval, 
                         feval=feval)



        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits


        gain = clf.feature_importance('gain')
        fold_importance_df = pd.DataFrame({'feature':clf.feature_name(),
                                           'split':clf.feature_importance('split'),
                                           'gain':100*gain/gain.sum(),
                                           'fold':n_fold,                        
                                           }).sort_values('gain',ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        #result = sklearn_mertric(valid_y, oof_preds[valid_idx])
        result = clf.best_score['valid']['l1']
        print('Fold %2d macro-f1 : %.6f' % (n_fold + 1, result))
        cv_resul.append(round(result,5))
        gc.collect()
        
    #score = np.array(cv_resul).mean()
    score = 'fee_pred'
    if USE_KFOLD:
        train_df[target] = oof_preds
        test_df[target] = sub_preds
        print(test_df[target].mean())
        train_df[target] = oof_preds
        #train_df[target] = train_df[target].map(label2current_service)
        test_df[target] = sub_preds
        #test_df[target] = test_df[target].map(label2current_service)
        print('all_cv', cv_resul)
        train_df[['user_id', target]].to_csv('./sub/val_{}.csv'.format(score), index= False)
        test_df[['user_id', target]].to_csv('./sub/sub_{}.csv'.format(score), index= False)
        print("test_df mean:")
    
    display_importances(feature_importance_df,score)



def display_importances(feature_importance_df_,score):
    ft = feature_importance_df_[["feature", "split","gain"]].groupby("feature").mean().sort_values(by="gain", ascending=False)
    print(ft.head(60))
    ft.to_csv('importance_lightgbm_{}.csv'.format(score),index=True)
    cols = ft[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]


####################################计算#################################################################


params = {
    'metric': 'mae',
    'boosting_type': 'gbdt', 
    'objective': 'regression',
    'feature_fraction': 0.65,
    'learning_rate': 0.1,
    'bagging_fraction': 0.65,
    #'bagging_freq': 2,
    'num_leaves': 64,
    'max_depth': -1, 
    'num_threads': 32, 
    'seed': 2018, 
    'verbose': -1,
    #'is_unbalance':True,
    }


categorical_columns = [
    'contract_type', 
#    'is_mix_service', 
#    'is_promise_low_consume', 
    'net_service',
    'gender']
for feature in categorical_columns:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()    
    train[feature] = encoder.fit_transform(train[feature].astype(str))    

#no_use_from_pandas = pd.read_csv('importance_lightgbm_0.903734.csv')
#x = list(no_use_from_pandas[no_use_from_pandas.gain==0.0]['feature'])
x = []
no_use = ['current_service', 'user_id','group','1_total_fee',
 
] + x

                                         
# for i in categorical_columns:
#     result.append(pd.get_dummies(train[[i]], columns= [i], dummy_na= False))

categorical_columns = []
all_data_frame = []
all_data_frame.append(train)

for aresult in result:
    all_data_frame.append(aresult)
    
train = concat(all_data_frame)

feats = [f for f in train.columns if f not in no_use]
categorical_columns = [f for f in categorical_columns if f not in no_use]

train['1_total_fee'] = train['1_total_fee'].astype(float)
#train_old = train.iloc[shape1:shape2]
#train = train.iloc[:shape1]
#4000`:
clf = kfold_lightgbm(params,train,feats,'1_total_fee' ,5 , num_boost_round=4000, categorical_features=categorical_columns)#2000
