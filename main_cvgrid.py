import numpy as np
import pandas as pd
import datetime
import random
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

data = pd.read_csv("~/HE ML3/Data/train.csv")
trainc, testc = train_test_split(data,test_size=0.6,random_state=131,stratify=data[target])

train_cvall = list()
cvfold_error = list()
cvfold_estop = list()
vtrain = trainc

for i in range(5,1,-1):
	vtrain, ltrain = train_test_split(vtrain,test_size=(1/float(i)),random_state=11,stratify=vtrain[target])
	train_cvall.append(ltrain)
train_cvall.append(vtrain)

for i in range(5):
	train = pd.DataFrame()
	for j in range(5):
		if (j!= i):
			train = pd.concat([train,train_cvall[j]],axis=0)
	test = train_cvall[i]
	train = data_prep(train)
	test = data_prep(test)
	train = one_hot(train,cdcols)
	test = one_hot(test,cdcols)
	train = leave_oneout_enc_train(train,looecols,target,countval)
	test = leave_oneout_enc_test(test,train,looecols,target,countval)
	predictors = [x for x in train.columns if x not in 
	['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click']]
	norm_var = ['siteid_category','siteid_merchant','siteid_countrycode','siteid_offerid','category_merchant','category_countrycode',
			'category_offerid','merchant_countrycode','merchant_offerid','countrycode_offerid','cc_merchant','cc_category',
			'cc_siteid','cc_offerid']
	scaler = MinMaxScaler()
	train[norm_var] = scaler.fit_transform(train[norm_var])
	test[norm_var] = scaler.fit_transform(test[norm_var])
	dtrain = xgb.DMatrix(data=train[predictors].values,label=train[target].values)
	dverify = xgb.DMatrix(data=test[predictors].values,label=test[target].values)
		traineval = {}
	for spw in range(1,27,5):
		params = {'learning_rate':0.1,'n_estimators':24,'max_depth':4,'min_child_weight':7,'gamma':0,'subsample':0.6,
			  	  'silent':False,'colsample_bytree':0.7,'objective':'binary:logistic','nthread':16,'scale_pos_weight':11,
			  	  'reg_alpha':100}
		params['eval_metric'] = 'auc'
		evallist = [(dverify,'CrossValidation')]
		xgb_mod = xgb.train(params,dtrain,num_boost_round=24,evals=evallist,early_stopping_rounds=50,evals_result=traineval,verbose_eval=True)
		cvfold_error.append([spw,traineval['CrossValidation']['auc'][len(traineval['CrossValidation']['auc'])-1]])
	
print cvfold_error
cv_error_auc = pd.DataFrame(cvfold_error)
cv_error_auc.columns=['scale_pos_weight','cvauc']
cv_error_auc = pd.pivot_table(cv_error_auc,values='cvauc',index=['scale_pos_weight'],aggfunc=np.mean)
cv_error_auc.to_csv('~/HE ML3/Others/XGB_Tune_SPW.csv')
print cv_error_auc

trainc = trainc


