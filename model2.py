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
#from utils2 import *
#from utils3 import *
from datetime import datetime
from datetime import timedelta
#from tqdm import tqdm
#test

warnings.simplefilter(action='ignore', category=FutureWarning)


USE_KFOLD = True

data_path = './input/'

####################################读入文件####################################################
#要准备hzs的两个get most文件
def astype(x,t):
    try:
        return t(x)
    except:
        return np.nan

def have_0(x):
    try:
        r = x.split('.')[1][-1]
        return 0 if r=='0' else 1
    except:
        return 1

str_dict = {'1_total_fee': 'str',
 '2_total_fee': 'str',
 '3_total_fee': 'str',
 '4_total_fee': 'str',
 'pay_num': 'str',
 }


have_0_c = ['1_total_fee',
'2_total_fee',
'3_total_fee',
'4_total_fee',
'pay_num']

def deal(data):
    for c in have_0_c:
        data['have_0_{}'.format(c)] = data[c].apply(have_0)
        try:
            data[c] = data[c].astype(float)
        except:
            pass
    data['2_total_fee'] = data['2_total_fee'].apply(lambda x: astype(x,float))
    data['3_total_fee'] = data['3_total_fee'].apply(lambda x: astype(x,float))
    data['age'] = data['age'].apply(lambda x: astype(x,int))
    data['gender'] = data['gender'].apply(lambda x: astype(x,int))
    data.loc[data['age']==0,'age'] = np.nan
    data.loc[data['1_total_fee'] < 0, '1_total_fee'] = np.nan
    data.loc[data['2_total_fee'] < 0, '2_total_fee'] = np.nan
    data.loc[data['3_total_fee'] < 0, '3_total_fee'] = np.nan
    data.loc[data['4_total_fee'] < 0, '4_total_fee'] = np.nan
    for c in [
    '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
    'month_traffic', 'last_month_traffic', 'local_trafffic_month',
    'local_caller_time', 'service1_caller_time', 'service2_caller_time',
    'many_over_bill', 'contract_type', 'contract_time', 'pay_num', ]:
        data[c] = data[c].round(4)
    return data

train = pd.read_csv(data_path + 'train.csv',dtype=str_dict)
train = deal(train)
train.drop_duplicates(subset = ['1_total_fee','2_total_fee','3_total_fee',
 'month_traffic','pay_times','last_month_traffic','service2_caller_time','age'],inplace=True)
train = train[train['current_service'] != 999999]
test = pd.read_csv(data_path + 'test.csv',dtype=str_dict)
test = deal(test)

train_old = pd.read_csv('./input/train_old.csv',dtype=str_dict)[:]
train_old = deal(train_old)
train_old.drop_duplicates(subset = ['1_total_fee','2_total_fee','3_total_fee',
 'month_traffic','pay_times','last_month_traffic','service2_caller_time','age'],inplace=True)
    


print(len(train))


label2current_service =dict(zip(range(0,len(set(train['current_service']))),sorted(list(set(train['current_service'])))))
current_service2label =dict(zip(sorted(list(set(train['current_service']))),range(0,len(set(train['current_service'])))))
print(len(label2current_service))
train = train.append(test).reset_index(drop = True)
print(len(train))
shape1 = len(train)
train['is_b'] = 1
train_old['is_b'] = 0      
train = train.append(train_old).reset_index(drop = True)
print(len(train))
shape2 = len(train)

get_most = pd.read_csv('Magic_Feature_Exclude_Old.csv')
get_most2 = pd.read_csv('Magic_Feature_Include_Old.csv')

####################################特征工程###################################################

call_time = ['local_caller_time', 'service1_caller_time', 'service2_caller_time']
traffic = ['month_traffic','last_month_traffic','local_trafffic_month']
cat_cols = ['service_type','contract_type', 'net_service', 'gender', 'complaint_level',
               #3              #9,8           #4             #3         #4           
   'is_mix_service',  'many_over_bill', 'is_promise_low_consume',   #2    
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
#


train['fea-min'] = train[[str(1+i) +'_total_fee' for i in range(4)]].min(axis = 1)

for column in ['1_total_fee', '2_total_fee', '3_total_fee',  '4_total_fee', 'fea-min']:
    get_most.columns = [column,column+'_most']
    train = train.merge(get_most,on=column,how='left')
    
for column in ['1_total_fee', '2_total_fee', '3_total_fee',  '4_total_fee', 'fea-min']:
    get_most2.columns = [column,column+'_most2']
    train = train.merge(get_most2,on=column,how='left')

for column in ['1_total_fee', '2_total_fee', '3_total_fee',  '4_total_fee', 'pay_num','fea-min']:
    train[column+'_int'] = train[column].fillna(-1).astype('int')
    train[column+'_int_last'] = train[column+'_int']%10 #last int 
    train[column+'_decimal'] = round(((train[column]-train[column+'_int'])*100).fillna(-1)).astype('int')    #decimal
    train[column+'_decimal_is_0'] = (train[column+'_decimal']==0).astype('int')
    train[column+'_decimal_is_5'] = (train[column+'_decimal']%5==0).astype('int') 
    train[column+'_decimal_last'] = train[column+'_decimal']%10
    train[column+'_decimal_last2'] = train[column+'_decimal']//5 
    train[column+'_extra_fee'] = ((train[column]*100)-600)%1000
    train[column+'_27perMB'] = ((train[column+'_extra_fee']%27 == 0)&(train[column+'_extra_fee'] != 0)).astype('int')
    train[column+'_15perMB'] = ((train[column+'_extra_fee']%15 == 0)&(train[column+'_extra_fee'] != 0)).astype('int')
    train = one_hot_encoder(train,column,n=2000,nan_as_category=True)
train['pay_num_last2'] = train['pay_num_int']%100  
train['former_complaint_fee_last2'] = round(train['former_complaint_fee'])%100      
    

train['4-fea-dealta'] = round((train['4_total_fee'] - train['3_total_fee'])*100).fillna(999999.9).astype('int')
train['3-fea-dealta'] = round((train['3_total_fee'] - train['2_total_fee'])*100).fillna(999999.9).astype('int')
train['2-fea-dealta'] = round((train['2_total_fee'] - train['1_total_fee'])*100).fillna(999999.9).astype('int')
train['1-fea-dealta'] = round((train['4_total_fee'] - train['1_total_fee'])*100).fillna(999999.9).astype('int')  
train['1-3-fea-dealta'] = round((train['3_total_fee'] - train['1_total_fee'])*100).fillna(999999.9).astype('int') 
train['1-min-fea-dealta'] = round((train['1_total_fee'] - train['fea-min'])*100).fillna(999999.9).astype('int') 
for column in ['4-fea-dealta', '3-fea-dealta', '2-fea-dealta', '1-fea-dealta','1-3-fea-dealta','1-min-fea-dealta']:
    train[column+'_is_0'] = (train[column]==0).astype('int')
    train[column+'_is_6000'] = ((train[column]%6000 == 0)&(train[column] != 0)).astype('int') 
    train[column+'_is_5'] = ((train[column]%5 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_10'] = ((train[column]%10 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_15'] = ((train[column]%15 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_27'] = ((train[column]%27 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_30'] = ((train[column]%30 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_50'] = ((train[column]%50 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_100'] = ((train[column]%100 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_500'] = ((train[column]%500 == 0)&(train[column] != 0)).astype('int')

for column in ['month_traffic', 'last_month_traffic', 'local_trafffic_month']:
    train[column+'_is_int'] = ((train[column]%1 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_512'] = ((train[column]%512 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_50'] = ((train[column]%50 == 0)&(train[column] != 0)).astype('int')
    train[column+'_is_double'] = ((train[column]%512%50 == 0)&(train[column] != 0)&(train[column+'_is_512'] == 0)&(train[column+'_is_50'] == 0)).astype('int')
    train = one_hot_encoder(train,column,n=2000,nan_as_category=True)
    
train['service12'] = train['service2_caller_time']+train['service1_caller_time']
for column in ['local_caller_time', 'service1_caller_time', 'service2_caller_time','service12']:
    train[column+'_decimal'] =  round(((round(train[column])- train[column])*60)).astype('int')
    train[column+'_decimal_is_int'] = ((train[column+'_decimal']==0)&(train[column] != 0)).astype('int')

train = one_hot_encoder(train,'online_time',n=5000,nan_as_category=True)
train = one_hot_encoder(train,'contract_time',n=5000,nan_as_category=True) 

print(train.shape)
train = one_hot_encoder(train,'contract_type',n=1,nan_as_category=True) 



#lable 映射 
train['current_service'] = train['current_service'].map(current_service2label)


train['age'] = train['age'].fillna(-20)
train['cut_age'] = train['age'].apply(lambda x: int(x/10))
train['cut_online'] = (train['online_time'] / 12).astype(int)



train['4-fea-dealta'] = train['4_total_fee'] - train['3_total_fee']
train['3-fea-dealta'] = train['3_total_fee'] - train['2_total_fee']
train['2-fea-dealta'] = train['2_total_fee'] - train['1_total_fee']
train['1-fea-dealta'] = train['4_total_fee'] - train['1_total_fee']

train['4-fea-dealta_'] = train['4_total_fee'] / (train['3_total_fee']+0.00001)
train['3-fea-dealta_'] = train['3_total_fee'] / (train['2_total_fee']+0.00001)
train['2-fea-dealta_'] = train['2_total_fee'] / (train['1_total_fee']+0.00001)
train['1-fea-dealta_'] = train['4_total_fee'] / (train['1_total_fee']+0.00001)
train['pay_num-dealta_'] = train['pay_num'] / (train['1_total_fee']+0.00001)



train['month_traffic_delata'] = train['month_traffic'] - train['last_month_traffic']
train['month_traffic_delata_'] = train['month_traffic'] / (train['last_month_traffic']+0.00001)
train['2month_traffic_sum'] = train['month_traffic'] + train['last_month_traffic']
train['add_month_traffic'] = train['month_traffic'] - train['local_trafffic_month']
train['add_month_traffic_'] = train['month_traffic'] / (train['local_trafffic_month']+0.00001)

train['service1_caller_time_delata'] = train['service1_caller_time'] / (train['service2_caller_time']+0.00001)
train['service1_caller_time_delata2'] = train['service1_caller_time'] / (train['local_caller_time']+0.00001)
train['service2_caller_time_delata_'] = train['service2_caller_time'] / (train['local_caller_time']+0.00001)
train['local_caller_time_reatio'] = train['local_caller_time']/(train['service1_caller_time']+train['service2_caller_time']+0.00001)

train['div_online_time_contract'] = train['contract_time'] / (train['online_time']+0.00001)
train['div_online_time_contract'] = train['contract_time'] - train['online_time']


train['div_former_complaint_num'] = train['former_complaint_num'] / (train['pay_times']+0.00001)
train['div_former_complaint_num'] = train['former_complaint_num'] - train['pay_times']


train['fea-sum'] = train[[str(1+i) +'_total_fee' for i in range(4)]].sum(axis = 1)
train['fea-var'] = train[[str(1+i) +'_total_fee' for i in range(4)]].var(axis = 1)
train['fea-max'] = train[[str(1+i) +'_total_fee' for i in range(4)]].max(axis = 1)
train['fea-min'] = train[[str(1+i) +'_total_fee' for i in range(4)]].min(axis = 1)
train['fea-mean4'] = train[[str(1+i) +'_total_fee' for i in range(4)]].sum(axis = 1)
train['fea-mean3'] = train[[str(1+i) +'_total_fee' for i in range(3)]].sum(axis = 1)
train['fea-mean2'] = train[[str(1+i) +'_total_fee' for i in range(2)]].sum(axis = 1)
train['fea-extra'] = train['fea-sum']-4*train['fea-min']
train['1_total_fee_extra_for_min'] = train['1_total_fee']-train['fea-min']
train['fea_unum'] = train[['1_total_fee','2_total_fee','3_total_fee', '4_total_fee']].nunique(axis=1)

train['call_time_sum'] = train[call_time].sum(axis = 1)
train['call_time_var'] = train[call_time].var(axis = 1)
train['call_time_min'] = train[call_time].min(axis = 1)
train['call_time_max'] = train[call_time].max(axis = 1)

train['traffic_sum'] = train[traffic].sum(axis = 1)
train['traffic_var'] = train[traffic].var(axis = 1)
train['traffic_min'] = train[traffic].min(axis = 1)
train['traffic_max'] = train[traffic].max(axis = 1)


train['average_pay'] = train['pay_num'] / train['pay_times']


train['div_traffic_price_2'] = train['last_month_traffic']/ 1000 / train['2_total_fee']
train['div_traffic_price_3']  = train['local_trafffic_month']/ 1000 / train['1_total_fee']
train['div_add_month_traffic_price']  = train['add_month_traffic']/ 1000 / train['1_total_fee']
train['div_local_caller_time_price']  = train['local_trafffic_month'] / 1000/ train['1_total_fee']


train['1-min-fea-dealta_div'] = train['1-min-fea-dealta']/(train['service1_caller_time']+0.0001)
train['div_service1_caller_time_price']  = train['service1_caller_time'] / train['1_total_fee']
train['div_local_caller_time']  = train['local_caller_time'] / train['1_total_fee']
train['div_call_time_sum_price']  = train['call_time_sum'] / train['1_total_fee']
train['1_total_fee_maybe_real_calller'] = train['1_total_fee']- train['service1_caller_time']*0.15
train['1_total_fee_maybe_real_calller2'] = train['1_total_fee']- train['service1_caller_time']*0.1
train['1_total_fee_extra_for_min_caller_time'] = train['1_total_fee_extra_for_min']/(train['service1_caller_time']+0.001)

train['div_service1_caller_time'] = train['service1_caller_time']/train['last_month_traffic']
train['div_local_caller_time'] = train['local_caller_time']/train['last_month_traffic']
train['div_local_caller_time2'] = train['local_caller_time']/train['month_traffic']


train['avg_complain_fee'] = train['former_complaint_fee'] / (train['former_complaint_num'] + 0.000000001)


result = []

result.append(get_feat_ngroup(train,['cut_age','gender']))
for size_feat in ['1_total_fee','2_total_fee','3_total_fee', '4_total_fee','pay_num',
'last_month_traffic','month_traffic','local_trafffic_month',
 'local_caller_time','service1_caller_time','service2_caller_time']:
    result.append(get_feat_size(train,[size_feat]))
    
    
result.append(get_feat_stat_feat(train, ['contract_type'], ['1_total_fee'], ['max']))
result.append(get_feat_stat_feat(train, ['contract_type'], ['2_total_fee'], ['mean']))
result.append(get_feat_stat_feat(train, ['contract_type'], ['last_month_traffic'], ['var','mean']))
result.append(get_feat_stat_feat(train, ['contract_type'], ['call_time_sum'], ['mean']))

for base_feat in [['contract_type']]:
    for other_feat in ['1_total_fee',  'pay_num',
                         'month_traffic', 'last_month_traffic', 'local_trafffic_month', 
                         'local_caller_time', 'service1_caller_time', 'service2_caller_time',
                       ]:
        stat_list = ['mean']
        tmp = get_feat_stat_feat(train,base_feat,[other_feat],stat_list=stat_list)
        name = tmp.columns[0]
        train[name] = tmp
        train[name+'_comp'] = train[other_feat].values-train[name].values


train['1_total_fee_ratio'] = train['1_total_fee']/(train['fea-sum']+0.000001)
train['3_total_fee_ratio'] = train['3_total_fee']/(train['fea-sum']+0.000001)
train['call_time_sum_ratio'] = train['call_time_sum']/(train['traffic_sum']+0.000001) 
train['call_time_sum_ratio2'] = train['call_time_sum']/(train['fea-sum']+0.000001) 
train['traffic_sum_ratio1'] = train['traffic_sum']/(train['fea-sum']+0.000001) 

####################################lgb和metric,post函数###################################################
    
def gen_top1():
    def replace_white_user_id(submission):
        white_user_id = pd.read_csv('white.csv')
        white_dict = dict(zip(white_user_id['user_id'],white_user_id['current_service'].astype(int)))
        submission['current_service'] = list(map(lambda x,y: y if x not in white_dict else white_dict[x],submission['user_id'],submission['current_service']))
        return submission
    sub = pd.read_csv('./sub/sub_model_2.csv')
    sub = replace_white_user_id(sub)
    sub.to_csv('./sub_top1.csv',index=None)


def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    preds = np.argmax(preds.reshape(11, -1),axis=0)
    score_vali = f1_score(y_true=labels,y_pred=preds,average='macro')
    return 'macro_f1_score', score_vali, True

def evaluate_macroF1_lgb(data_vali, preds):  
    labels = data_vali.astype(int)
    preds = np.array(preds)
    preds = np.argmax(preds,axis=1)
    score_vali = f1_score(y_true=labels,y_pred=preds,average='macro')
    return  score_vali

def kfold_lightgbm(params,df, predictors,target,num_folds, stratified = True,
                   objective='', metrics='',debug= False,
                   feval = f1_score_vali, early_stopping_rounds=100, num_boost_round=100, verbose_eval=50, categorical_features=None,sklearn_mertric = evaluate_macroF1_lgb ):

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
    oof_preds = np.zeros((train_df.shape[0],11))
    sub_preds = np.zeros((test_df.shape[0],11))
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

        train_x = pd.concat([train_x,train_old[feats]])
        train_y = pd.concat([train_y,train_old[target]])

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
#                         feval=feval
                         )



        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration)/ folds.n_splits


        gain = clf.feature_importance('gain')
        fold_importance_df = pd.DataFrame({'feature':clf.feature_name(),
                                           'split':clf.feature_importance('split'),
                                           'gain':100*gain/gain.sum(),
                                           'fold':n_fold,                        
                                           }).sort_values('gain',ascending=False)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        result = evaluate_macroF1_lgb(valid_y, oof_preds[valid_idx])
#        result = clf.best_score['valid']['macro_f1_score']
        print('Fold %2d macro-f1 : %.6f' % (n_fold + 1, result))
        cv_resul.append(round(result,5))
        gc.collect()
        
    #score = np.array(cv_resul).mean()
    score = 'model_2'
    if USE_KFOLD:
        #print('Full f1 score %.6f' % score)
        for i in range(11):
            train_df["class_" + str(i)] = oof_preds[:,i]
            test_df["class_" + str(i)] = sub_preds[:,i]
        train_df[['user_id'] + ["class_" + str(i) for i in range(11)]].to_csv('./cv/val_prob_{}.csv'.format(score), index= False, float_format = '%.4f')
        test_df[['user_id'] + ["class_" + str(i) for i in range(11)]].to_csv('./cv/sub_prob_{}.csv'.format(score), index= False, float_format = '%.4f')   
        oof_preds = [np.argmax(x)for x in oof_preds]
        sub_preds = [np.argmax(x)for x in sub_preds]    
        train_df[target] = oof_preds
        test_df[target] = sub_preds
        print(test_df[target].mean())
        train_df[target] = oof_preds
        train_df[target] = train_df[target].map(label2current_service)
        test_df[target] = sub_preds
        test_df[target] = test_df[target].map(label2current_service)
        print('all_cv', cv_resul)
        train_df[['user_id', target]].to_csv('./sub/val_{}.csv'.format(score), index= False)
        test_df[['user_id', target]].to_csv('./sub/sub_{}.csv'.format(score), index= False)
        print("test_df mean:")
    
    display_importances(feature_importance_df,score)
    #gen_top1()




def display_importances(feature_importance_df_,score):
    ft = feature_importance_df_[["feature", "split","gain"]].groupby("feature").mean().sort_values(by="gain", ascending=False)
    print(ft.head(60))
    ft.to_csv('importance_lightgbm_{}.csv'.format(score),index=True)
    cols = ft[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]


####################################计算#################################################################


params = {
    'metric': 'multi_logloss',
    'num_class':11,
    'boosting_type': 'gbdt', 
    'objective': 'multiclass',
    'feature_fraction': 0.7,
    'learning_rate': 0.02,
    'bagging_fraction': 0.7,
    #'bagging_freq': 2,
    'num_leaves': 64,
    'max_depth': -1, 
    'num_threads': 16, 
    'seed': 2018, 
    'verbose': -1,
    #'is_unbalance':True,
    }


categorical_columns = [
    'contract_type', 
    'net_service',
    'gender']
for feature in categorical_columns:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()    
    train[feature] = encoder.fit_transform(train[feature].astype(str))    


x = []
no_use = ['current_service', 'user_id','group',
 
] + x

                                         


categorical_columns = []
all_data_frame = []
all_data_frame.append(train)

for aresult in result:
    all_data_frame.append(aresult)
    
train = concat(all_data_frame)
feats = [f for f in train.columns if f not in no_use]
categorical_columns = [f for f in categorical_columns if f not in no_use]

train_old = train.iloc[shape1:shape2]
train = train.iloc[:shape1]
#train = train[train.service_type!=1]
#train_old = train_old[train_old.service_type!=1]
clf = kfold_lightgbm(params,train,feats,'current_service' ,5 , num_boost_round=4000, categorical_features=categorical_columns)
