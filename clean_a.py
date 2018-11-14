from tool import *

data_path = './input/'
d = {89950166: 1, 89950167: 2, 89950168: 5, 90063345: 0, 90109916: 4,
 90155946: 8, 99999825: 10, 99999826: 7, 99999827: 6, 99999828: 3, 99999830: 9}
rd = {0: 90063345, 1: 89950166, 2: 89950167, 3: 99999828, 4: 90109916,
 5: 89950168, 6: 99999827, 7: 99999826, 8: 90155946, 9: 99999830, 10: 99999825}


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
 'last_month_traffic': 'str',
 'local_caller_time': 'str',
 'local_trafffic_month': 'str',
 'month_traffic': 'str',
 'pay_num': 'str',
 'service1_caller_time': 'str',
 'service2_caller_time': 'str'}
train = pd.read_csv(data_path + 'train_old.csv',dtype=str_dict)
#test = pd.read_csv(data_path + 'republish_test.csv',dtype=str_dict)
train['label'] = train['current_service'].map(d)

have_0_c = ['1_total_fee',
'2_total_fee',
'3_total_fee',
'4_total_fee',
'month_traffic',
'last_month_traffic',
'local_trafffic_month',
'local_caller_time',
'service1_caller_time',
'service2_caller_time',
'pay_num']
def deal(data):
    for c in have_0_c:
        data['have_0_{}'.format(c)] = data[c].apply(have_0)
        try:
            data[c] = data[c].astype(float)
        except:
            pass
    for c in ['1_total_fee','2_total_fee', '3_total_fee', '4_total_fee','pay_num' ]:
        data['{}_len'.format(c)] = data[c].astype(str).apply(lambda x: 0 if '.' not in x else len(x.split('.')[1]))
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

train = deal(train)
#test = deal(test)


data_path='./data/a/'
train.to_csv(data_path + 'train.csv',index=False)
#test.to_csv(data_path + 'test.csv',index=False)
#print('预处理完成')











