import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import random
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test,modelfit

random.seed(138727)
data = pd.read_csv("./Data/train.csv")
train, test = train_test_split(data,test_size=0.2,random_state=131)

cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant']
target = 'click'

train = data_prep(train)
test = data_prep(test)
train = one_hot(train,cdcols)
test = one_hot(test,cdcols)
train = leave_oneout_enc_train(train,looecols,'click')
test = leave_oneout_enc_test(test,train,looecols,'click')

remove_var = ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid']
predictors = [x for x in train.columns if x not in remove_var]

xgb_csf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=6,min_child_weight=1,gamma=0,subsample=0.8,
	colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=138727)
xgb_mod = modelfit(xgb_csf,train,predictors)

test_predictions = xgb_mod.predict(test[predictors])
test_predprob = xgb_mod.predict_proba(test[predictors])[:,1]
print "\nModel Report"
print "Accuracy : %.4g" % accuracy_score(test[target].values,test_predictions)
print "AUC Score (Train): %f" % roc_auc_score(test[target],test_predprob)