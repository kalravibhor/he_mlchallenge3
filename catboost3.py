#catboost original

# This script will introduce you to use catboost package in #python. Set the variables<br />
# in the define variables cell and run the code<br />
# 

# In[4]:


###deleting the workspace 
#get_ipython().magic(u'reset')

##importing the libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import os
import glob
from catboost import Pool
from catboost import cv

#import rpy2

###custom functions
#from functions_gg import one_hot


# In[5]:


##define variables
dsloc="" ##location of the datasets on your system
yvarname="click"  ##y var name in your dataset
nthread=16

# You can install this package using: `pip install catboost`

# In[6]:


##input datasets
train = pd.read_csv(dsloc+"train.csv")
test = pd.read_csv(dsloc+"test.csv")

print "Data input done"
# In[7]:


#train=train.sample(12137000)


# In[9]:


# check missing values per column
#train.isnull().sum(axis=0)/train.shape[0]


# In[10]:


# impute missing values,can we do better?
train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)


# In[11]:


# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])


# In[12]:


# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute


# In[13]:


#train.to_csv('/home/gautam/Desktop/hackerearth_ds/train_prep.csv')
#test.to_csv('/home/gautam/Desktop/hackerearth_ds/test_prep.csv')


# In[14]:


## convert the numeric fields into categorical, to be read by catboost
cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')

#all categorical columns
cat_cols = cols + ['countrycode','browserid','devid']


# In[15]:


### convert categorical variables from text to 0,1
for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values) + list(test[col].values))
    train[col] = lbl.transform(list(train[col].values))
    test[col] = lbl.transform(list(test[col].values))

print "Categorical variables encoded into numbers"
# In[16]:


### use the columns for model
cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))


###validation dataset to be used finally to track the actual model performance
###stratified sampling to create training and validations sets

#valid=train[train['datetime'] > '2017-01-11 00:00:00']
#train=train[train['datetime'] <= '2017-01-11 00:00:00']





# In[23]:


### split the data into train and evaluation for the catboost model
#X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.3,stratify=trainY)

valid=train[train['datetime'] > '2017-01-18 00:00:00']
train=train[train['datetime'] <= '2017-01-18 00:00:00']

####training dataset
X_train = train[cols_to_use]
y_train = train[yvarname]
X_test = valid[cols_to_use]
y_test = valid[yvarname]


###specifying the model paramters
model = CatBoostClassifier(depth=6, iterations=500, learning_rate=0.03, 
 eval_metric='AUC', rsm=1, auto_stop_pval=0.01, use_best_model = True, thread=nthread, verbose=True)
#model1 = CatBoostClassifier(depth=10,iterations=30, learning_rate=0.1, eval_metric='AUC', random_seed=1, thread=nthread, verbose=True)

if False:'''
CatBoostClassifier(iterations=500, 
                         learning_rate=0.03, 
                         depth=6, 
                         l2_leaf_reg=3, 
                         rsm=1, 
                         loss_function='Logloss',
                         border=None,
                         border_count=32,
                         feature_border_type='MinEntropy',
                         fold_permutation_block_size=1,
                         auto_stop_pval=0,
                         gradient_iterations=None,
                         leaf_estimation_method=None,
                         thread_count=None,
                         random_seed=None,
                         use_best_model=False,
                         verbose=False,
                         ctr_description=None,
                         ctr_border_count=5050,
                         max_ctr_complexity=4,
                         priors=None,
                         has_time=False,
                         name='experiment',
                         ignored_features=None,
                         train_dir=None,
                         custom_loss=None,
                         eval_metric=None,
                         class_weights=None)
'''


# In[25]:



# In[26]:


print train.head()
#cat_cols

#catboost accepts categorical variables as indexes
cat_cols = [0,1,2,3,4,5,6,7,8,9]

print cols_to_use
print "Using categorical columns as:",cat_cols


# In[27]:

print "Model fitting process starts now"
###model fitting using catboost
fit=model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,verbose =True
          ,plot=False)



# In[ ]:


###understanding the model built, diagnosis
#fit.feature_importances
#model.feature_importances(X_train,y=y_train,cat_features=None,weight=None,baseline=None,thread_count=4)


# In[28]:


##validation of the model
###auc of the validation dataset, it should mirror the leaderboard, the difference betweeen this value and the public leaderboard value would signify the variance error and thus tell, what are the chances of losing in the final submission
pred_valid = model.predict_proba(valid[cols_to_use],verbose=True)[:,1]
print roc_auc_score(np.array(valid[yvarname]), pred_valid)

# In[ ]:


###submission
pred = model.predict_proba(test[cols_to_use],verbose=True)[:,1]
sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv('submission_'+str(len([i for i in glob.glob('submission*.{}'.format('csv'))])+1)+'.csv',index=False)