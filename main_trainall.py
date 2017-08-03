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

cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant']
target = 'click'

random.seed(138727)
data = pd.read_csv("~/Data/train.csv")
train = data

train = data_prep(train)
train = one_hot(train,cdcols)
train = leave_oneout_enc_train(train,looecols,target)

predictors = [x for x in train.columns if x not in 
['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click']]

xgb_csf = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,silent=False,
	colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=138727)
xgb_mod = modelfit(xgb_csf,train,predictors,target)

try:
	print [xgb_mod.best_score,xgb_mod.best_iteration]
except:
	print "Early stopping not initiated"