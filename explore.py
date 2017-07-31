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
# datetime	timestamp
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
	table = table.reset_index()
	try:
		hh_rr_list = (table[table['positive_response_rate']>=res_cutoff_high][colname]).tolist()
	except:
		hh_rr_list = []
	try:
		lw_rr_list = (table[table['positive_response_rate']<=res_cutoff_low][colname]).tolist()
	except:
		lw_rr_list = []
	return [hh_rr_list,lw_rr_list]

df['devid'] = df['devid'].fillna('')
df['browserid'] = df['browserid'].fillna('')
df['datetime'] =  pd.to_datetime(df['datetime'])
df['dayofthweek'] = df['datetime'].dt.dayofweek
df['timeofday'] = df['datetime'].dt.time
df['timeofday'] = df['timeofday'].astype('str')
df['timeofday'] = df['timeofday'].apply(lambda x: (int(x.split(':')[0])*3600) + (int(x.split(':')[1])*60) + (int(x.split(':')[2]))) 

df.loc[df['browserid'].isin(['IE','Internet Explorer']),'browserid'] = 'InternetExplorer'
df.loc[df['browserid']=='Mozilla Firefox','browserid'] = 'Firefox'
df.loc[df['browserid']=='Google Chrome','browserid'] = 'Chrome'

# array(['', 'Chrome', 'Edge', 'Firefox', 'Google Chrome', 'IE',
#        'Internet Explorer', 'InternetExplorer', 'Mozilla',
#        'Mozilla Firefox', 'Opera', 'Safari'], dtype=object)
# # Average/Variance