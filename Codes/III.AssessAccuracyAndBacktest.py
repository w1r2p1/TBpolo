# III. Assess accuracy and backtest
'''Here we look at every model and seek for the most profitable one'''
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
################### (a) Load previously built datasets
trainset_final = pd.read_csv('./Data/TrainSet_final.csv')
trainset = pd.read_csv('./Data/TrainSet.csv')
testset_final = pd.read_csv('./Data/TestSet_final.csv')
testset = pd.read_csv('./Data/TestSet.csv')

stoploss = 0.02
takeprofit = 0.05
fees = 0.0009 # transaction fees : 0.09% for example


def simulate_expectancy(p, stoploss, takeprofit, fees = 0.0009):
    '''Compute earnings and loss with given fees, stoploss, takeprofit'''
    win = (1-fees)*(1+takeprofit)*(1-fees) -1
    loss = (1-fees)*(1-stoploss)*(1-fees) -1
    return(win, loss)

#################### (b) Basic strategy : pick the best model and bet on bullish trends over the testset
recap = pd.read_csv('Comparative_All_models.csv').sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]

with open("./Models/DL_model_{}PC.pkl".format(nPCs), 'rb') as f:
    clf = pk.load(f)
# Compute predictions on testset
testset['preds'] = (clf.predict(testset_final.iloc[:, :nPCs]) > 0.5)*1
# keep only the timesteps in which the model predicts a bullish trend
testset = testset[testset['preds'] == 1]

# Compute earnings column
a = simulate_expectancy(p = 0, stoploss, takeprofit, fees = )
trades['EarningsBullish'] = (testset['preds'] == testset['result'])*a[0] + (testset['preds'] != testset['result'])*a[1]

# Now plot our trading strategy
plt.plot(testset['date'], np.cumsum(testset['EarningsBullish']))
plt.title('ROI = {} %'.format(np.mean(testset['EarningsBullish'])))
plt.xlabel('Date')
plt.xlabel('Cumulative Earnings')





