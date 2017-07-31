from import numpy as np
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import random
from functions import data_prep,one_hot,leave_oneout_enc_train,leave_oneout_enc_test

random.seed(138727)
cdcols = ['devid','browserid','countrycode']
looecols = ['category','merchant']

train = pd.read_csv("/Users/Kronos/Desktop/HE ML3/Data/train.csv")
train = data_prep(train)
train = one_hot(train,cdcols)
train = leave_oneout_enc_train(train,looecols,'click')