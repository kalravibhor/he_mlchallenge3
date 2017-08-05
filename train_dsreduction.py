import numpy as np
import pandas as pd
import Density_Sampling
import random
from functions import data_prep,one_hot,leave_oneout_enc_train

cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant']
target = 'click'

random.seed(138727)
data = pd.read_csv("~/Data/train.csv")
train = data

train = data_prep(train)
train = one_hot(train,cdcols)
train = leave_oneout_enc_train(train,looecols,target)

remove_var = ['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click']
predictors = [x for x in train.columns if x not in remove_var]
train.loc[train['looe_merchant'].isnull(),'looe_merchant'] = 0

si_50p_red = Density_Sampling.density_sampling(train[predictors+['click']],metric='euclidean',desired_samples=int(train.shape[0]*0.5))
si_40p_red = Density_Sampling.density_sampling(train[predictors+['click']],metric='euclidean',desired_samples=int(train.shape[0]*0.4))
si_30p_red = Density_Sampling.density_sampling(train[predictors+['click']],metric='euclidean',desired_samples=int(train.shape[0]*0.3))

train_50pred = train.loc[si_50p_red,:]
train_40pred = train.loc[si_40p_red,:]
train_30pred = train.loc[si_30p_red,:]

train_50pred.to_csv("~/Others/train_reduced_50p.csv")
train_40pred.to_csv("~/Others/train_reduced_40p.csv")
train_30pred.to_csv("~/Others/train_reduced_30p.csv")