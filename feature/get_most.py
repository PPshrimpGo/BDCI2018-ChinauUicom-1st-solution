import pandas as pd
import numpy as np

data_path = '../input/'
output_path = '../'

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
	    'many_over_bill', 'contract_type', 'contract_time', 'pay_num', 
    ]:
        data[c] = data[c].round(6)
    return data

train = pd.read_csv(data_path + 'train.csv',dtype=str_dict)
train = deal(train)
train = train[train['current_service'] != 999999]
test = pd.read_csv(data_path + 'test.csv',dtype=str_dict)
test = deal(test)
train_old = pd.read_csv(data_path +'train_old.csv',dtype=str_dict)[:]
train_old = deal(train_old)    

def get_magic_feature(df, outname):
    """
    It is the magic beer and niaobu, try it and enjoy!
    """
    df['fea_unum'] = df[['1_total_fee','2_total_fee','3_total_fee', '4_total_fee']].nunique(axis=1)
    df.drop_duplicates(subset =['1_total_fee','2_total_fee','3_total_fee', '4_total_fee'],inplace=True)
    df = df[df.fea_unum>2]
    for month1_month2 in [
        [1,2],
        [1,3],
        [1,4],
        [2,1],
        [2,3],
        [2,4],
        [3,1],
        [3,2],
        [3,4],
        [4,1],
        [4,2],
        [4,3],
    ]:
        month1, month2 = str(month1_month2[0]), str(month1_month2[1])
        mstr = '_total_fee'
        tmp = df.groupby([month1 + mstr, month2 + mstr]).size().reset_index()
        tmp.columns =['first','second','{}_total_fee_{}_total_fee'.format(month1,month2)]
        if month1_month2 == [1,2]:
            result_df = tmp
        else:
            result_df = result_df.merge(tmp, on = ['first','second'], how = 'outer')

    tmpall = result_df
    tmpall = tmpall[tmpall.second!=0]
    tmpall['count'] =  tmpall.iloc[:,2:].sum(axis=1)
    tmpall = tmpall.merge(tmpall.groupby('second',as_index=False)['count'].agg({'sum':'sum'}),on='second',how='left')
    tmpall['rate'] = tmpall['count'] / tmpall['sum']
    tmpall = tmpall.sort_values(['first','rate'],ascending=False)
    tmpall =  tmpall [tmpall['count']>10]
    tmpall = tmpall.sort_values(['first','count'],ascending=False)
    tmp_res = tmpall.drop_duplicates('first',keep='first')
    tmp_res[tmp_res['count']>10].to_csv(output_path + outname, columns = ['first','count'],index = False)
    
# Magic_Feature_Exclude_Old
train = train.append(test).reset_index(drop = True)
train.drop_duplicates(subset = ['1_total_fee','2_total_fee','3_total_fee',
 'month_traffic','pay_times','last_month_traffic','service2_caller_time','age'],inplace=True)
get_magic_feature(train, 'Magic_Feature_Exclude_Old.csv')

# Magic_Feature_Include_Old
train = train.append(train_old).reset_index(drop = True)
train.drop_duplicates(subset = ['1_total_fee','2_total_fee','3_total_fee',
 'month_traffic','pay_times','last_month_traffic','service2_caller_time','age'],inplace=True)
get_magic_feature(train, 'Magic_Feature_Include_Old.csv')
