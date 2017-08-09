import numpy as np
import pandas as pd
import datetime
import random
from glob import glob
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split,GridSearchCV	
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test

random.seed(138727)
cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant','offerid','siteid']
target = 'click'
countval = 'ID'
norm_var = ['siteid_category','siteid_merchant','siteid_countrycode','siteid_offerid','category_merchant','category_countrycode',
			'category_offerid','merchant_countrycode','merchant_offerid','countrycode_offerid','cc_merchant','cc_category',
			'cc_siteid','cc_offerid']
predictors = ['dayofthweek', 'timeofday', 'siteid_category', 'siteid_merchant', 'siteid_countrycode', 'siteid_offerid', 
			  'category_merchant', 'category_countrycode', 'category_offerid', 'merchant_countrycode', 'merchant_offerid',
			  'countrycode_offerid', 'devid_Desktop', 'devid_Mobile', 'devid_Tablet', 'browserid_Chrome', 'browserid_Edge',
			  'browserid_Firefox', 'browserid_InternetExplorer', 'browserid_Opera', 'countrycode_b', 'countrycode_c',
			  'countrycode_d', 'countrycode_e', 'countrycode_f', 'cc_category', 'looe_category', 'cc_merchant', 'looe_merchant',
			  'cc_offerid', 'looe_offerid', 'cc_siteid', 'looe_siteid']

data = pd.read_csv("~/HE ML3/Data/train.csv")
test_data = pd.read_csv("~/HE ML3/Data/test.csv")
trainc, testc = train_test_split(data,test_size=0.6,random_state=131,stratify=data[target])

train_cvall = list()
cvfold_error = list()
cvfold_estop = list()
pred = list()
scaler = MinMaxScaler()
vtrain = trainc

testprep = data_prep(test_data)
testprep = one_hot(testprep,cdcols)
testprep = leave_oneout_enc_test(testprep,data,looecols,target,countval)
testprep[norm_var] = scaler.fit_transform(testprep[norm_var])

data = data_prep(data)
data = one_hot(data,cdcols)
data = leave_oneout_enc_train(data,looecols,target,countval)
data[norm_var] = scaler.fit_transform(data[norm_var])

dtrain = xgb.DMatrix(data=data[predictors].values,label=data[target].values)
dtest = xgb.DMatrix(testprep[predictors].values)
params = {'learning_rate':0.1,'n_estimators':24,'max_depth':4,'min_child_weight':7,'gamma':0,'subsample':0.6,
	  	  'silent':False,'colsample_bytree':0.7,'objective':'binary:logistic','nthread':16,'scale_pos_weight':11,
	  	  'reg_alpha':100,'eval_metric':'auc'}
xgb_mod_all = xgb.train(params,dtrain,num_boost_round=24)
pred_all = xgb_mod_all.predict(dtest)
sub01 = pd.DataFrame({'ID':testprep['ID'],'click':pred_all})
sub01.to_csv("~/HE ML3/Others/submission_01.csv",index=False)

for i in range(5,1,-1):
	vtrain, ltrain = train_test_split(vtrain,test_size=(1/float(i)),random_state=11,stratify=vtrain[target])
	train_cvall.append(ltrain)

train_cvall.append(vtrain)

for i in range(5):
	train = pd.DataFrame()
	for j in range(5):
		if (j!= i):
			train = pd.concat([train,train_cvall[j]],axis=0)
	train = data_prep(train)
	train = one_hot(train,cdcols)
	train = leave_oneout_enc_train(train,looecols,target,countval)
	train[norm_var] = scaler.fit_transform(train[norm_var])
	dtrain = xgb.DMatrix(data=train[predictors].values,label=train[target].values)
	params = {'learning_rate':0.1,'n_estimators':24,'max_depth':4,'min_child_weight':7,'gamma':0,'subsample':0.6,
		  	  'silent':False,'colsample_bytree':0.7,'objective':'binary:logistic','nthread':16,'scale_pos_weight':11,
		  	  'reg_alpha':100}
	params['eval_metric'] = 'auc'
	xgb_mod = xgb.train(params,dtrain,num_boost_round=24)
	pred.append(xgb_mod.predict(dtest))

pred_avg = (pred[0] + pred[1] + pred[2] + pred[3] + pred[4])/5
sub02 = pd.DataFrame({'ID':testprep['ID'],'click':pred_avg})
sub02.to_csv("~/HE ML3/Others/submission_02.csv",index=False)
