import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import random
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test

random.seed(138727)
cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant']
target = 'click'

data = pd.read_csv("~/HE ML3/Data/train.csv")
trainc, testc = train_test_split(data,test_size=0.6,random_state=131,stratify=data[target])

train_cvall = list()
cvfold_error = list()
cvfold_estop = list()

for i in [5,4,3,2]:
	trainc, ltrain = train_test_split(trainc,test_size=(1/float(i)),random_state=11,stratify=trainc[target])
	train_cvall.append(ltrain)
train_cvall.append(trainc)

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
	train = leave_oneout_enc_train(train,looecols,target)
	test = leave_oneout_enc_test(test,train,looecols,target)
	predictors = [x for x in train.columns if x not in 
	['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click']]
	norm_var = ['siteid_category','siteid_merchant','siteid_countrycode','siteid_offerid','category_merchant','category_countrycode',
			'category_offerid','merchant_countrycode','merchant_offerid','countrycode_offerid','cc_merchant','cc_category',
			'cc_siteid','cc_offerid']
	scaler = MinMaxScaler()
	train[norm_var] = scaler.fit_transform(train[norm_var])
	params = {'learning_rate':0.1,'n_estimators':1000,'max_depth':4,'min_child_weight':1,'gamma':0,'subsample':0.8,'silent':False,
						'colsample_bytree':0.8,'objective':'binary:logistic','nthread':16,'scale_pos_weight':1}
	dtrain = xgb.DMatrix(data=train[predictors].values,label=train[target].values)
	dverify = xgb.DMatrix(data=test[predictors].values,label=test[target].values)
	evallist = [(dverify,'CrossValidation')]
	traineval = {}
	xgb_mod = xgb.train(params,dtrain,num_boost_round=1000,evals=evallist,early_stopping_rounds=50,evals_result=traineval,verbose_eval=True)
	cvfold_error.append(traineval['CrossValidation']['error'])
	
cv_error_auc = pd.DataFrame(cvfold_error)
cv_error_auc = cv_error_auc.transpose()
cv_error_auc = cv_error_auc.fillna(0)
cv_error_auc.columns=['auc_cvf_01','auc_cvf_02','auc_cvf_03','auc_cvf_04','auc_cvf_05']
cv_error_auc['mean_auc'] = (cv_error_auc['auc_cvf_01'] + cv_error_auc['auc_cvf_02'] + cv_error_auc['auc_cvf_03']
							 + cv_error_auc['auc_cvf_04'] + cv_error_auc['auc_cvf_05'])/5
cv_error_auc.to_csv('~/Others/CrossValidation_AUCError.csv')