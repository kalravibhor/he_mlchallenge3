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
norm_var = ['siteid_category','siteid_merchant','siteid_countrycode','siteid_offerid','category_merchant','category_countrycode',
			'category_offerid','merchant_countrycode','merchant_offerid','countrycode_offerid','cc_merchant','cc_category',
			'cc_siteid','cc_offerid']
predictors = ['dayofthweek', 'timeofday', 'siteid_category', 'siteid_merchant', 'siteid_countrycode', 'siteid_offerid', 
			  'category_merchant', 'category_countrycode', 'category_offerid', 'merchant_countrycode', 'merchant_offerid',
			  'countrycode_offerid', 'devid_Desktop', 'devid_Mobile', 'devid_Tablet', 'browserid_Chrome', 'browserid_Edge',
			  'browserid_Firefox', 'browserid_InternetExplorer', 'browserid_Opera', 'countrycode_b', 'countrycode_c',
			  'countrycode_d', 'countrycode_e', 'countrycode_f', 'cc_category', 'looe_category', 'cc_merchant', 'looe_merchant',
			  'cc_offerid', 'looe_offerid', 'cc_siteid', 'looe_siteid']
param_type_dict = {'n_estimators':int,
				   'max_depth':int,
				   'min_child_weight':int,
				   'learning_rate':float,
				   'gamma':float,
				   'subsample':float,
				   'colsample_bytree':float,
				   'scale_pos_weight':int}

data = pd.read_csv("~/HE ML3/Data/train.csv")
trainc, testc = train_test_split(data,test_size=0.6,random_state=131,stratify=data[target])
vtrain = trainc

train_cvall = list()
cvfold_error = list()
cvfold_estop = list()
scaler = MinMaxScaler()
precision = 0.01

for i in range(5,1,-1):
	vtrain, ltrain = train_test_split(vtrain,test_size=(1/float(i)),random_state=11,stratify=vtrain[target])
	train_cvall.append(ltrain)
train_cvall.append(vtrain)

def create_grid(param_list,param_range,global_param_list):
	cvgrid = pd.DataFrame()
	if (len(param_list)>1):
		tprange = param_range[0]
		tpname = param_list[0]
		param_list.pop(0)
		param_range.pop(0)
		df = create_grid(param_list,param_range,global_param_list)
		df.columns = global_param_list
		for i in tprange:
			df[tpname] = i
			cvgrid = pd.concat([cvgrid,df],axis=0)
	else:
		sparse = [0]*len(global_param_list)
		param_idx = global_param_list.index(param_list[0])
		for j in param_range[0]:
			sparse[param_idx] = j
			dfrow = pd.Series(sparse,index=global_param_list)
			cvgrid = cvgrid.append(dfrow,ignore_index=True)
	return cvgrid

def xgbtunecv(ldf,param_list,param_range,cv_folds,itrno,prevopt_results):
	prescolname = param_list[:]
	cvgrid = create_grid(param_list,param_range,param_list)
	cvgrid = cvgrid[prescolname]
	cvgrid = cvgrid.reset_index(drop=True)
	for colname in list(cvgrid):
		cvgrid[colname] = cvgrid[colname].astype(param_type_dict[colname])
	for i in range(cv_folds):
		train = pd.DataFrame()
		for j in range(cv_folds):
			if (j!= i):
				train = pd.concat([train,ldf[j]],axis=0)
		test = ldf[i]
		train = data_prep(train)
		test = data_prep(test)
		train = one_hot(train,cdcols)
		test = one_hot(test,cdcols)
		train = leave_oneout_enc_train(train,looecols,target,countval)
		test = leave_oneout_enc_test(test,train,looecols,target,countval)
		train[norm_var] = scaler.fit_transform(train[norm_var])
		test[norm_var] = scaler.fit_transform(test[norm_var])
		dtrain = xgb.DMatrix(data=train[predictors].values,label=train[target].values)
		dverify = xgb.DMatrix(data=test[predictors].values,label=test[target].values)
		params = {'learning_rate':0.1,'n_estimators':500,'max_depth':5,'min_child_weight':1,'gamma':0,'subsample':0.8,'eval_metric':'auc',
			  	  'silent':False,'colsample_bytree':0.8,'objective':'binary:logistic','nthread':16,'scale_pos_weight':1}
		traineval = {}
		evallist = [(dverify,'CrossValidation')]
		try:
			for prev_opt_list in prevopt_results:
				popmname = prev_opt_list[0]
				popmvalue = prev_opt_list[1]
				params[popmname] = popmvalue
		if (param_list != ['n_estimators']):
			for index,row in cvgrid.iterrows():
				cvelist = list()
				for colname in cvgrid.columns:
					params[colname] = row[colname]
				xgb_mod = xgb.train(params,dtrain,num_boost_round=params['n_estimators'],evals=evallist,
									evals_result=traineval,verbose_eval=True)
				cvelist = cvgrid.loc[index,:].tolist()
				cvelist.extend([traineval['CrossValidation']['auc'][len(traineval['CrossValidation']['auc'])-1]])
				cvfold_error.append(cvelist)
		else:
			params['n_estimators'] = param_range[0][-1]
			xgb_mod = xgb.train(params,dtrain,num_boost_round=params['n_estimators'],evals=evallist,early_stopping_rounds=50,
								evals_result=traineval,verbose_eval=True)
			cvfold_error.append(traineval['CrossValidation']['auc'])
	cv_error_auc = pd.DataFrame(cvfold_error)
	if (param_list == ['n_estimators']):
		cv_error_auc = cv_error_auc.transpose()
		cv_error_auc = cv_error_auc.reset_index(drop=False)
	cv_error_auc = cv_error_auc.fillna(0)
	tmpcolname = list(cvgrid)
	tmpcolname.extend(['cvauc'])
	cv_error_auc.columns=tmpcolname
	cv_error_auc = pd.pivot_table(cv_error_auc,values='cvauc',index=list(cvgrid),aggfunc=np.mean)
	cv_error_auc = cv_error_auc.reset_index()
	print cv_error_auc
	cv_error_auc.to_csv('~/HE ML3/Others/XGB_autotuning_results_0'+ itrno + '.csv')
	cv_error_auc = cv_error_auc.sort_values(by='cvauc',ascending=False)
	cv_error_auc = cv_error_auc.reset_index(drop=True)
	return cv_error_auc.loc[0,cv_error_auc.columns != 'cvauc'].tolist()

pvy_opt_prm_list = list()

niters = xgbtunecv(train_cvall,['n_estimators'],[range(1,250,1)],5,1,pvy_opt_prm_list)
niters = int(niters[0])
print "Due to early stopping, number of iterations fixed at : " + str(niters)
pvy_opt_prm_list.append(['n_estimators',niters])

mdp_01,mcw_01 = xgbtunecv(train_cvall,['max_depth','min_child_weight'],[range(4,11,2),range(1,8,2)],5,pvy_opt_prm_list)
print "First iteration resulted in max_depth at : " + str(mdp_01) + " and min_child_weight at : " + str(mcw_01)
mdp_02,mcw_02 = xgbtunecv(train_cvall,['max_depth','min_child_weight'],[range(mdp-1,mdp+2,1),range(mcw-1,mcw+2,1)],5,pvy_opt_prm_list)
print "Second iteration resulted in max_depth at : " + str(mdp_02) + " and min_child_weight at : " + str(mcw_02)
pvy_opt_prm_list.append(['max_depth',mdp_02])
pvy_opt_prm_list.append(['min_child_weight',mcw_02])

gamma_01 = xgbtunecv(train_cvall,['gamma'],[i/10.0 for i in range(6)],5,1,pvy_opt_prm_list)
gamma_01 = int(gamma_01[0])
print "First iteration resulted in gamma at : " + str(gamma_01)
gamma_02 = xgbtunecv(train_cvall,['gamma'],[i/100.0 for i in range(int(gamma_01*100)-5,int(gamma_01*100)+6,2)],5,1,pvy_opt_prm_list)
gamma_02 = int(gamma_02[0])
print "Second iteration resulted in gamma at : " + str(gamma_02)
pvy_opt_prm_list.append(['gamma',gamma_02])