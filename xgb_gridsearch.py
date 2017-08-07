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
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test,modelfit

random.seed(138727)
cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant','offerid','siteid']
target = 'click'
countval = 'ID'

data = pd.read_csv("~/HE ML3/Data/train.csv")
train, test = train_test_split(data,test_size=0.6,random_state=131,stratify=data[target])

train = data_prep(train)
train = one_hot(train,cdcols)
train = leave_oneout_enc_train(train,looecols,target,countval)

predictors = [x for x in train.columns if x not in ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click']]
norm_var = ['siteid_category','siteid_merchant','siteid_countrycode','siteid_offerid','category_merchant','category_countrycode',
'category_offerid','merchant_countrycode','merchant_offerid','countrycode_offerid','cc_merchant','cc_category',
'cc_siteid','cc_offerid']
scaler = MinMaxScaler()
train[norm_var] = scaler.fit_transform(train[norm_var])

xgb_csf = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=7,min_child_weight=1,silent=False,gamma=0.1,n_threads=16,
						subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',scale_pos_weight=1,random_state=27)
xgb_mod = modelfit(xgb_csf,train,predictors,target)

# param_md_mcw = {
#  'gamma':[i/10.0 for i in range(0,5)]
# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1,n_estimators=80,max_depth=7,min_child_weight=1,silent=False,
# 						gamma=0.1,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',scale_pos_weight=1,random_state=27),
# 						param_grid=param_md_mcw,scoring='roc_auc',n_jobs=16,iid=False,cv=5)
# gsearch1.fit(train[predictors],train[target])
# print gsearch1.grid_scores_
# print gsearch1.best_params_
# print gsearch1.best_score_
# print gsearch1.cv_results_