import numpy as np
import pandas as pd

train = pd.read_csv("/Users/Kronos/Desktop/HE ML3/Data/train.csv")
len(np.unique(train[""]))

records : 12137810
category : 271
site : 1431688
offerid : 847510
merchant : 697
countrycode : 6

def statmeasures_catvar(var_name,count_top,count_bot):


def one_hot(df,cols):
	for each in cols:
		dummies = pd.get_dummies(df[each], prefix=each, drop_first=True)
		df = pd.concat([df, dummies], axis=1)
	return df

pd.crosstab(index=df[each],columns='count')

cdcols = ['devid','browserid','countrycode']

# Variable	Description
# ID	Unique ID
# datetime	timestamp
# siteid	website id
# offerid	offer id (commission based offers)
# category	offer category
# merchant	seller ID
# countrycode	country where affiliates reach is present
# browserid	browser used
# devid	device used
# click	target variable