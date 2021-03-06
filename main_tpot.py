import numpy as np
import pandas as pd
import datetime
from tpot import TPOTClassifier
import random
from sklearn.model_selection import train_test_split
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test

cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant']
target = 'click'

random.seed(138727)
data = pd.read_csv("~/HE ML3/Data/train.csv")
train, test = train_test_split(data,test_size=0.2,random_state=31,stratify=data[target])

train = data_prep(train)
train = one_hot(train,cdcols)
train = leave_oneout_enc_train(train,looecols,target)
test = data_prep(test)
test = one_hot(test,cdcols)
test = leave_oneout_enc_test(test,train,looecols,target)

predictors = [x for x in train.columns if x not in 
['ID','datetime','siteid','offerid','category','merchant','countrycode','browserid','devid','click']]

tpot = TPOTClassifier(generations=5,population_size=20,verbosity=2)
tpot.fit(train[predictors],train[target])
print(tpot.score(test[predictors],test[target]))
tpot.export('~/HE ML3/Codes/tpot_pipeline.py')