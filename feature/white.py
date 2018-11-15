import pandas as pd
import numpy as np
pd.set_option('max_columns',1000) 
pd.set_option('max_row',300) 

input_path = '../input/'
output_path = '../'
train = pd.read_csv(input_path + 'train.csv')
test = pd.read_csv(input_path +'test.csv')

def astype(x,t):
    try:
        return t(x)
    except:
        return np.nan

def deal(data):

    data['2_total_fee'] = data['2_total_fee'].apply(lambda x: astype(x,float))
    data['3_total_fee'] = data['3_total_fee'].apply(lambda x: astype(x,float))
    data['age'] = data['age'].apply(lambda x: astype(x,int))
    data['gender'] = data['gender'].apply(lambda x: astype(x,int))
    for c in [
        '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
        'month_traffic', 'last_month_traffic', 'local_trafffic_month',
        'local_caller_time', 'service1_caller_time', 'service2_caller_time',
        'many_over_bill', 'contract_type', 'contract_time', 'pay_num', 
        ]:
        data[c] = data[c].round(4)
    return data

train = deal(train)
train = train[train['current_service'] != 999999]
test = deal(test)

cc = [
    '1_total_fee',
    '2_total_fee',
    '3_total_fee',
    'month_traffic',
    'pay_times',
    'last_month_traffic',
    'service2_caller_time',
    'age'
]

train = train.drop_duplicates(cc)
white = test.merge(train,on=cc,how='left')
white = white[~white['current_service'].isnull()][['user_id_x','user_id_y','current_service']].copy()
white['current_service'] = white['current_service'].astype('int')
white.drop(['user_id_y'],inplace=True,axis=1)
white.columns=['user_id','current_service']
white.to_csv(output_path + 'white.csv', index = False)
#white = pd.read_csv('white.csv')
