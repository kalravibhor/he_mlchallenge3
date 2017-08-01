# One hot encoding for categorical variables with low cardinality
def one_hot(df,cols):
	for each in cols:
		dummies = pd.get_dummies(df[each], prefix=each, drop_first=True)
		df = pd.concat([df, dummies], axis=1)
	return df

# Leave one out encoding for categorical variables with high cardinality (train)
def leave_oneout_enc_train(df,colnames,target):
	for colname in colnames:
		noise_mean = 0
		noise_std = np.std(df[target])
		cfq = pd.crosstab(index=df[colname],columns='cc_' + colname)
		srr = pd.pivot_table(df,values=target,index=[colname],aggfunc=np.sum)
		srr.columns = ['sum_rr']
		df = df.set_index(colname)
		df = df.merge(cfq.merge(srr,how='left',left_index=True,right_index=True),how='left',left_index=True,right_index=True)
		df['looe_' + colname] = (df['sum_rr'] - df[target])/(df['cc_' + colname] - 1) + (np.random.normal(loc=noise_mean,scale=noise_std,size=(df.shape[0]))/100).tolist()
		df = df.drop(['sum_rr'],axis=1)
		df = df.reset_index()
	return df

# Leave one out encoding for categorical variables with high cardinality (test)
def leave_oneout_enc_test(df_test,df_train,colnames,target):
	for colname in colnames:
		cfq = pd.crosstab(index=df_train[colname],columns='cc_' + colname)
		srr = pd.pivot_table(df_train,values=target,index=[colname],aggfunc=np.sum)
		srr.columns = ['sum_rr']
		df_test = df_test.set_index(colname)
		df_test = df_test.merge(cfq.merge(srr,how='left',left_index=True,right_index=True),how='left',left_index=True,right_index=True)
		df_test['looe_' + colname] = (df_test['sum_rr']/df_test['cc_' + colname])
		df_test = df_test.drop(['sum_rr'],axis=1)
		df_test = df_test.reset_index()
	return df_test

# Data pre-processing
def data_prep(df):
	df['devid'] = df['devid'].fillna('')
	df['browserid'] = df['browserid'].fillna('')
	df['siteid'] = df['devid'].fillna(0)
	df.loc[df['browserid'].isin(['IE','Internet Explorer']),'browserid'] = 'InternetExplorer'
	df.loc[df['browserid']=='Mozilla Firefox','browserid'] = 'Firefox'
	df.loc[df['browserid']=='Google Chrome','browserid'] = 'Chrome'
	df['datetime'] =  pd.to_datetime(df['datetime'])
	df['dayofthweek'] = df['datetime'].dt.dayofweek
	df['timeofday'] = df['datetime'].dt.time
	df['timeofday'] = df['timeofday'].astype('str')
	df['timeofday'] = df['timeofday'].apply(lambda x: (int(x.split(':')[0])*3600) + (int(x.split(':')[1])*60) + (int(x.split(':')[2])))
	return df

# Response distribution for categorical variables
def res_dist(df,colname,target,idval):
	cfq = pd.crosstab(index=df[colname],columns='count')
	freq_cutoff = np.mean(cfq['count'])
	res_cutoff_high = np.mean(df[target]) + np.std(df[target])
	res_cutoff_low = np.mean(df[target]) - np.std(df[target])
	table = pd.pivot_table(df,values=idval,index=[colname],columns=[target],aggfunc='count',margins=True)
	table = table.fillna(0)
	table.columns = ['negative','positive','all']
	table['positive_response_rate'] = table['positive']/table['all']
	table = table[table['all'] > freq_cutoff]
	try:
		hh_rr_list = table[table['positive_response_rate']>=res_cutoff_high][colname]
	except:
		hh_rr_list = []
	try:
		lw_rr_list = table[table['positive_response_rate']<=res_cutoff_low][colname]
	except:
		lw_rr_list = []
	return [hh_rr_list,lw_rr_list]

# Model fitting and summary
def modelfit(alg,dtrain,predictors,useTrainCV=True,cv_folds=5,early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
    return alg