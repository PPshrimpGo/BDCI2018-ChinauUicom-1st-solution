from tool import *

cache_path = '/'
inplace = False

############################### 工具函数 ###########################
# 合并节约内存
def concat(L):
    result = None
    for l in L:
        if result is None:
            result = l
        else:
            result[l.columns.tolist()] = l
    return result

# 统计转化率
def bys_rate(data,cate,cate2,label):
    temp = data.groupby(cate2,as_index=False)[label].agg({'count':'count','sum':'sum'}).rename(columns={'2_total_fee':'1_total_fee'})
    temp['rate'] = temp['sum']/temp['count']
    data_temp = data[[cate]].copy()
    data_temp = data_temp.merge(temp[[cate,'rate']],on=cate,how='left')
    return data_temp['rate']

# 统计转化率
def mul_rate(data,cate,label):
    temp1 = data.groupby([cate,label],as_index=False).size().unstack().fillna(0)
    temp2 = data.groupby([cate], as_index=False).size()
    temp2.loc[temp2 < 20] = np.nan
    temp3 = (temp1.T/temp2).T
    temp3.columns = [cate+'_'+str(c)+'_conversion' for c in temp3.columns]
    temp3 = temp3.reset_index()
    data = data.merge(temp3,on=cate,how='left')
    return data

# 相同的个数
def get_same_count(li):
    return pd.Series(li).value_counts().values[0]

# 相同的个数
def get_second_min(li):
    return sorted(li)[1]

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True, min_count=100,inplace=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    result = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in result.columns if c not in original_columns]
    cat_columns = [c for c in original_columns if c not in result.columns]
    if not inplace:
        for c in cat_columns:
            result[c] = df[c]
    for c in new_columns:
        if (result[c].sum()<100) or ((result.shape[0]-result[c].sum())<100):
            del result[c]
            new_columns.remove(c)
    return result, new_columns

# 连续特征离散化
def one_hot_encoder_continus(df, col, n_scatter=10,nan_as_category=True):
    df[col+'_scatter'] = pd.qcut(df[col],n_scatter)
    result = pd.get_dummies(df, columns=[col+'_scatter'], dummy_na=nan_as_category)
    return result

############################### 预处理函数 ###########################
def pre_treatment(data,data_key):
    result_path = cache_path + 'data_{}.feature'.format(data_key)
    if os.path.exists(result_path) & 0:
        data = pd.read_feature(result_path)
    else:
        month_fee = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
        data['total_fee_mean4'] = data[month_fee[:4]].mean(axis=1)
        data['total_fee_mean3'] = data[month_fee[:3]].mean(axis=1)
        data['total_fee_mean2'] = data[month_fee[:2]].mean(axis=1)
        data['total_fee_std4'] = data[month_fee[:4]].std(axis=1)
        # data['total_fee_mode4'] = data[month_fee[:4]].apply(mode,axis=1)
        data['total_fee_Standardization'] = data['total_fee_std4'] / (data['total_fee_mean4'] + 0.1)
        data['1_total_fee_rate12'] = data['1_total_fee'] / (data['2_total_fee'] + 0.1)
        data['1_total_fee_rate23'] = data['2_total_fee'] / (data['3_total_fee'] + 0.1)
        data['1_total_fee_rate34'] = data['3_total_fee'] / (data['4_total_fee'] + 0.1)
        data['1_total_fee_rate24'] = data['total_fee_mean2'] / (data['total_fee_mean4'] + 0.1)
        data['total_fee_max4'] = data[month_fee[:4]].max(axis=1)
        data['total_fee_min4'] = data[month_fee[:4]].min(axis=1)
        data['total_fee_second_min4'] = data[month_fee[:4]].apply(get_second_min,axis=1)
        data['service_caller_time_diff'] = data['service2_caller_time'] - data['service1_caller_time']
        data['service_caller_time_sum'] = data['service2_caller_time'] + data['service1_caller_time']
        data['service_caller_time_min'] = data[['service1_caller_time','service2_caller_time']].min(axis=1)
        data['service_caller_time_max'] = data[['service1_caller_time', 'service2_caller_time']].max(axis=1)

        data['1_total_fee_last0_number'] = count_encoding(data['1_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
        data['1_total_fee_last1_number'] = count_encoding(data['1_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
        data['1_total_fee_last2_number'] = count_encoding(data['1_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
        data['1_total_fee_last3_number'] = count_encoding(data['1_total_fee'].fillna(-1)//10)
        data['2_total_fee_last0_number'] = count_encoding(data['2_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
        data['2_total_fee_last1_number'] = count_encoding(data['2_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
        data['2_total_fee_last2_number'] = count_encoding(data['2_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
        data['2_total_fee_last3_number'] = count_encoding(data['2_total_fee'].fillna(-1) // 10)
        data['3_total_fee_last0_number'] = count_encoding(data['3_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
        data['3_total_fee_last1_number'] = count_encoding(data['3_total_fee'].fillna(-1).apply( lambda x:('%.2f' % x)[-2]).astype(int))
        data['3_total_fee_last2_number'] = count_encoding(data['3_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
        data['3_total_fee_last3_number'] = count_encoding(data['3_total_fee'].fillna(-1) // 10)
        data['4_total_fee_last0_number'] = count_encoding(data['4_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-1]).astype(int))
        data['4_total_fee_last1_number'] = count_encoding(data['4_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-2]).astype(int))
        data['4_total_fee_last2_number'] = count_encoding( data['4_total_fee'].fillna(-1).apply(lambda x: ('%.2f' % x)[-4]).astype(int))
        data['4_total_fee_last3_number'] = count_encoding(data['4_total_fee'].fillna(-1) // 10)
        # data['total_fee_sample_count'] = data[month_fee].apply(get_same_count,axis=1)

        for fee in ['1_total_fee','2_total_fee', '3_total_fee', '4_total_fee']:
            data['{}_1'.format(fee)] = ((data[fee] % 1==0) & (data[fee] !=0))
            data['{}_01'.format(fee)] = ((data[fee] % 0.1==0) & (data[fee] !=0))

        data['pay_number_last_2'] = data['pay_num']*100%100
        # data = one_hot_encoder_continus(data,'1_total_fee',20)
        # data = one_hot_encoder_continus(data, '2_total_fee', 20)
        # data = one_hot_encoder_continus(data, '3_total_fee', 20)
        # data = one_hot_encoder_continus(data, '4_total_fee', 20)
        # data = one_hot_encoder_continus(data, 'age', 10)
        # data = one_hot_encoder_continus(data, 'online_time', 10)

        data['1_total_fee_log'] = np.log(data['1_total_fee']+2)
        data['2_total_fee_log'] = np.log(data['2_total_fee'] + 2)
        data['3_total_fee_log'] = np.log(data['3_total_fee'] + 2)
        data['4_total_fee_log'] = np.log(data['4_total_fee'] + 2)
        data = grp_standard(data, 'contract_type', ['1_total_fee_log'], drop=False)
        data = grp_standard(data, 'contract_type', ['service_caller_time_min'], drop=False)
        data = grp_standard(data, 'contract_type', ['service_caller_time_max'], drop=False)
        data = grp_standard(data, 'contract_type', ['online_time'], drop=False)
        data = grp_standard(data, 'contract_type', ['age'], drop=False)
        data = grp_standard(data, 'net_service', ['1_total_fee_log'], drop=False)
        data = grp_standard(data, 'net_service', ['service_caller_time_min'], drop=False)
        data = grp_standard(data, 'net_service', ['service_caller_time_max'], drop=False)
        data = grp_standard(data, 'net_service', ['online_time'], drop=False)
        data = grp_standard(data, 'net_service', ['age'], drop=False)
        data['age_scatter'] = pd.qcut(data['age'], 5)
        data = grp_standard(data, 'age_scatter', ['1_total_fee_log'], drop=False)
        data = grp_standard(data, 'age_scatter', ['service_caller_time_min'], drop=False)
        data = grp_standard(data, 'age_scatter', ['service_caller_time_max'], drop=False)
        data = grp_standard(data, 'age_scatter', ['online_time'], drop=False)
        data = grp_standard(data, 'age_scatter', ['age'], drop=False)
        data['online_time_scatter'] = pd.qcut(data['online_time'], 5)
        data = grp_standard(data, 'online_time_scatter', ['1_total_fee_log'], drop=False)
        data = grp_standard(data, 'online_time_scatter', ['service_caller_time_min'], drop=False)
        data = grp_standard(data, 'online_time_scatter', ['service_caller_time_max'], drop=False)
        data = grp_standard(data, 'online_time_scatter', ['online_time'], drop=False)
        data = grp_standard(data, 'online_time_scatter', ['age'], drop=False)
        data = grp_standard(data, 'service_type', ['1_total_fee_log'], drop=False)
        data = grp_standard(data, 'service_type', ['service_caller_time_min'], drop=False)
        data = grp_standard(data, 'service_type', ['service_caller_time_max'], drop=False)
        data = grp_standard(data, 'service_type', ['online_time'], drop=False)
        data = grp_standard(data, 'service_type', ['age'], drop=False)

        del data['1_total_fee_log'],data['2_total_fee_log'],data['3_total_fee_log'],data['4_total_fee_log'], \
            data['age_scatter'],data['online_time_scatter']

        # data['online_time_count'] = count_encoding(data['online_time']//3)
        data['month_traffic_last_month_traffic_sum'] = data['month_traffic'] + data['last_month_traffic']
        data['month_traffic_last_month_traffic_diff'] = data['month_traffic'] - data['last_month_traffic']
        data['month_traffic_last_month_traffic_rate'] = data['month_traffic'] / (data['last_month_traffic']+0.01)
        data['outer_trafffic_month'] = data['month_traffic'] - data['local_trafffic_month']
        data['local_trafffic_month_month_traffic_rate'] = data['local_trafffic_month'] / (data['month_traffic'] + 0.01)

        data['month_traffic_last_month_traffic_sum_1_total_fee_rate'] = data['month_traffic_last_month_traffic_sum'] / (data['1_total_fee'] + 0.01)
        data['month_traffic_local_caller_time'] = data['month_traffic'] / (data['local_caller_time'] + 0.01)
        data['pay_num_per'] = data['pay_num'] / (data['pay_times']+0.01)
        data['total_fee_mean4_pay_num_rate'] = data['pay_num'] / (data['total_fee_mean4'] + 0.01)
        data['local_trafffic_month_spend'] = data['local_trafffic_month'] - data['last_month_traffic']
        data['month_traffic_1_total_fee_rate'] = data['month_traffic'] / (data['1_total_fee'] + 0.01)

        for traffic in ['month_traffic','last_month_traffic', 'local_trafffic_month']:
            data['{}_1'.format(traffic)] = ((data[traffic] % 1==0) & (data[traffic] !=0))
            data['{}_50'.format(traffic)] = ((data[traffic] % 50==0) & (data[traffic] !=0))
            data['{}_1024'.format(traffic)] = ((data[traffic] % 1024==0) & (data[traffic] !=0))
            data['{}_1024_50'.format(traffic)] = ((data[traffic] % 1024 % 50 == 0) & (data[traffic] != 0))

        data['service_caller_time'] = data['service1_caller_time'] + data['service2_caller_time']
        data['outer_caller_time'] = data['service_caller_time'] - data['local_caller_time']
        data['local_caller_time_rate'] = data['local_caller_time'] / (data['service_caller_time']+0.01)
        data['service1_caller_time_rate'] = data['service1_caller_time'] / (data['service_caller_time'] + 0.01)
        data['local_caller_time_service2_caller_time_rate'] = data['local_caller_time'] / (data['service2_caller_time'] + 0.01)
        data['service1_caller_time_1_total_fee_rate'] = data['service_caller_time'] / (data['1_total_fee'] + 0.01)

        # data['online_fee'] = groupby(data,data,'online_time','total_fee_mean4','median')
        # data['1_total_fee_10'] = data['1_total_fee']//10
        # data['1_total_fee_10_online_time'] = groupby(data, data, '1_total_fee_10', 'online_time', 'median')
        # del data['1_total_fee_10']
        # data['per_month_fee'] = data['pay_num'] / (data['online_time']+0.01)
        # data['per_month_times'] = data['pay_times'] / (data['online_time'] + 0.01)
        # data
        data['contract_time_count'] = count_encoding(data['contract_time'])
        data['pay_num_count'] = count_encoding(data['pay_num'])
        data['pay_num_last0_number'] = count_encoding(data['pay_num'].apply(lambda x: ('%.2f' % x)[-1]).astype(int))
        data['pay_num_last1_number'] = count_encoding(data['pay_num'].apply(lambda x: ('%.2f' % x)[-2]).astype(int))
        data['pay_num_last2_number'] = count_encoding(data['pay_num'].apply(lambda x: ('%.2f' % x)[-4]).astype(int))
        data['pay_num_count'] = count_encoding(data['pay_num'] // 10)
        data['age_count3'] = count_encoding(data['age'] // 3)
        data['age_count6'] = count_encoding(data['age'] // 6)
        data['age_count10'] = count_encoding(data['age'] // 10)
        # data['contract_time_count'] = count_encoding(data['contract_time'])
        # for i in range(11):
        #     data['temp'] = (data['label']==i).astype(int)
        #     data['1_total_fee_rate_cate{}'.format(i)] = cv_convert(data['1_total_fee'],data['temp'])
        # del data['temp']

        # data['1_total_fee_zheng'] = round(data['1_total_fee'])
        # data = one_hot_encoder(data, '1_total_fee_zheng', n=4000, nan_as_category=True)

        # 转化率
        data = mul_rate(data, 'pay_num', 'current_service')

        data = pd.get_dummies(data, columns=['contract_type'], dummy_na=-1)
        data = pd.get_dummies(data, columns=['net_service'], dummy_na=-1)
        data = pd.get_dummies(data, columns=['complaint_level'], dummy_na=-1)
        data.reset_index(drop=True,inplace=True)
        # data.to_feather(result_path)
    return data


############################### 特征函数 ###########################
# 特征
def get__feat(data,data_key):
    result_path = cache_path + '_feat_{}.feature'.format(data_key)
    if os.path.exists(result_path) & (not inplace):
        feat = pd.read_feature(result_path)
    else:
        data_temp = data.copy()

        feat.to_feather(result_path)
    return feat



# 二次处理特征
def second_feat(result):
    return result

def make_feat(data,data_key):
    t0 = time.time()
    # data_key = hashlib.md5(data.to_string().encode()).hexdigest()
    # #print('数据key为：{}'.format(data_key))
    result_path = cache_path + 'feat_set_{}.feature'.format(data_key)
    if os.path.exists(result_path) & 0:
        result = pd.read_feature(result_path, 'w')
    else:
        data = pre_treatment(data,'data_key')

        result = [data]
        # #print('开始构造特征...')
        # result.append(get_context_feat())     # context特征
        # result.append(get_user_feat())     # 用户特征
        # result.append(get_item_feat())     # 商品特征
        # result.append(get_shop_feat())     # 商店特征

        #print('开始合并特征...')
        result = concat(result)

        result = second_feat(result)

    #print('特征矩阵大小：{}'.format(result.shape))
    #print('生成特征一共用时{}秒'.format(time.time() - t0))
    return result













































