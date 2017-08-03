import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import random

random.seed(17)
train = pd.read_csv("/Users/Kronos/Desktop/HE ML3/Data/train.csv")
train = train.head(10000)
# train.dtypes
# len(np.unique(train[""]))

# records : 12137810
# category : 271
# site : 1431688
# offerid : 847510
# merchant : 697
# countrycode : 6

def one_hot(df,cols):
	for each in cols:
		dummies = pd.get_dummies(df[each], prefix=each, drop_first=True)
		df = pd.concat([df, dummies], axis=1)
	return df

ct = pd.crosstab(index=df[each],columns='count')
ct.sort_values(by=['col_0'],ascending=False)

cdcols = ['devid','browserid','countrycode']

# Variable	Description
# ID	Unique ID
# datetime	timestamp #the time when ad started displaying on affiliate's website.
# siteid	website id
# offerid	offer id (commission based offers)
# category	offer category
# merchant	seller ID
# countrycode	country where affiliates reach is present
# browserid	browser used
# devid	device used
# click	target variable

def leave_oneout_enc_train(df,colname,target):
	noise_mean = 0
	noise_std = np.std(df[target])
	cfq = pd.crosstab(index=df[colname],columns='cc_' + colname)
	srr = pd.pivot_table(df,values=target,index=[colname],aggfunc=np.sum)
	srr.columns = ['sum_rr']
	df = df.set_index(colname)
	df = df.join([cfq,srr],how='left')
	df['looe_' + colname] = (df['sum_rr'] - df[target])/(df['cc_' + colname] - 1) + (np.random.normal(loc=noise_mean,scale=noise_std,size=(df.shape[0]))/100).tolist()
	df = df.drop(['sum_rr'],axis=1)
	df = df.reset_index()
	return df

def leave_oneout_enc_test(df,colname,target):
	cfq = pd.crosstab(index=df[colname],columns='cc_' + colname)
	srr = pd.pivot_table(df,values=target,index=[colname],aggfunc=np.sum)
	srr.columns = ['sum_rr']
	df = df.set_index(colname)
	df = df.join([cfq,srr],how='left')
	df['looe_' + colname] = (df['sum_rr']/df['cc_' + colname])
	df = df.drop(['sum_rr'],axis=1)
	df = df.reset_index()
	return df

# Distribution of variables by response
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

df['devid'] = df['devid'].fillna('')
df['browserid'] = df['browserid'].fillna('')
df['datetime'] =  pd.to_datetime(df['datetime'])
df['dayofthweek'] = df['datetime'].dt.dayofweek
df['timeofday'] = df['datetime'].dt.time

df.loc[df['browserid'].isin(['IE','Internet Explorer']),'browserid'] = 'InternetExplorer'
df.loc[df['browserid']=='Mozilla Firefox','browserid'] = 'Firefox'
df.loc[df['browserid']=='Google Chrome','browserid'] = 'Chrome'

train.isnull().sum(axis=0)/train.shape[0]
# ID             0.000000
# datetime       0.000000
# siteid         0.099896
# offerid        0.000000
# category       0.000000
# merchant       0.000000
# countrycode    0.000000
# browserid      0.050118
# devid          0.149969
# click          0.000000

# xgb_mod = xgb_csf.fit(train[predictors],train[target],eval_metric='auc')
# test_predictions = xgb_mod.predict(test[predictors])
# test_predprob = xgb_mod.predict_proba(test[predictors])[:,1]
# print "Accuracy : %.4g" % accuracy_score(test[target].values,test_predictions)
# print "AUC Score : %f" % roc_auc_score(test[target],test_predprob)
# xgb_csf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,silent=False,
# 						colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=138727)
