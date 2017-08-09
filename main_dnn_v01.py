import numpy as np
import pandas as pd
import datetime
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV	
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test

random.seed(138727)
cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant','offerid','siteid']
target = 'click'
countval = 'ID'

data = pd.read_csv("~/Data/train.csv")
trainc, testc = train_test_split(data,test_size=0.6,random_state=131,stratify=data[target])

train_cvall = list()
vtrain = trainc
input_dim = 33

def def_kerasmodel():    
	model = Sequential()
    model.add(Dense(250,activation='relu',input_shape=(input_dim,)))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

callback = EarlyStopping(monitor='val_acc',patience=3)
model = def_kerasmodel(X_train)

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
try : 
	model.fit(train[predictors],train[target],batch_size=128,epochs=10,verbose=2,callbacks=[callback],
			  validation_data=(test[predictors],test[target]),shuffle=True)
except RuntimeError:
	print "Model not complied. Check configuration"
print model.summary()

vpreds = model.predict_proba(test[predictors])
roc_auc_score(y_true = Y_valid[:,1], y_score=vpreds)