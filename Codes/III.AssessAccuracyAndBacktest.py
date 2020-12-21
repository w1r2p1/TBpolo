# Article III : Assess accuracy and backtest
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
trainset_final = pd.read_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
trainset = pd.read_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

validation_set_final = pd.read_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
validation_set = pd.read_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

testset_final = pd.read_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
testset = pd.read_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

stoploss = 0.05
takeprofit = 0.1
fees = 0.00125 # transaction fees : 0.125% for example

list_nPCs = [10, 20, 30, 40]

def compute_earnings_loss(stoploss, takeprofit, fees):
    '''Compute earnings and loss with given fees, stoploss, takeprofit'''
    win = (1-fees)*(1+takeprofit)*(1-fees) -1
    loss = (1-fees)*(1-stoploss)*(1-fees) -1
    return(win, loss)

#################### (b) Basic strategy : pick the best model and bet on bullish trends over the testset
recap = pd.read_csv('Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]

with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    clf = pk.load(f)
# Compute predictions on testset
testset['preds'] = (clf.predict(testset_final.iloc[:, :nPCs]) > 0.5)*1
testset['proba1'] = clf.predict(testset_final.iloc[:, :nPCs])

# Compute earnings column
a = compute_earnings_loss(stoploss, takeprofit, fees)
testset['EarningsBullish'] = (testset['preds'] == testset['result'])*a[0] + (testset['preds'] != testset['result'])*a[1]

# keep only the timesteps in which the model predicts a bullish trend
testset1 = testset[testset['preds'] == 1].copy()

# Now plot our trading strategy
plt.plot(pd.to_datetime(testset1['date']), np.cumsum(testset1['EarningsBullish']))
plt.title('Best model over the testset \n ROI = {} %'.format(100*np.mean(testset1['EarningsBullish'])))
plt.xlabel('Date')
plt.xlabel('Cumulative Earnings')
plt.show()

# Display the entry points
plt.plot(pd.to_datetime(testset['date']), testset['close'])
plt.scatter(pd.to_datetime(testset1['date']), testset1['close'], c = (testset1['EarningsBullish']>0))
plt.title('Entry points \n Yellow = Win, Blue = Loss')
plt.show()



################### (c) More evolved strategy : look for the threshold to limit to the cases where p>a (for the best model)
def table_recap(df, stoploss, takeprofit, nPCs, columnA = 'proba1', columnB = 'EarningsBullish'):
    ''' Summarize the strategy by steps of 0.05, depending on which column (i.e. strategy)
    we choose'''
    recap = pd.DataFrame(np.zeros((int(10), 0)))
    recap['stoploss'] = stoploss
    recap['takeprofit'] = takeprofit
    recap['nPCs'] = nPCs
    recap['Min'] = [0.5 + k*0.05 for k in range(10)]
    recap['Max'] = 1
    recap['ROI%'] = 0
    recap['nTrades'] = 0
    for i in range(len(recap['Min'])):
        min, max = recap['Min'].iloc[i], recap['Max'].iloc[i]
        df2 = df[(df[columnA] > min) & (df[columnA] < max)]
        recap['ROI%'].iloc[i] = 100 * np.mean(df2[columnB])
        recap['nTrades'].iloc[i] = df2.shape[0]

    return(recap)

# Load best model
nPCs = list_nPCs[0]
with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    clf = pk.load(f)

# Compute predictions on validation_set
validation_set['preds'] = (clf.predict(validation_set_final.iloc[:, :nPCs]) > 0.5)*1
validation_set['proba1'] = clf.predict(validation_set_final.iloc[:, :nPCs])
a = compute_earnings_loss(stoploss, takeprofit, fees)
validation_set['EarningsBullish'] = (validation_set['preds'] == validation_set['result'])*a[0] + (validation_set['preds'] != validation_set['result'])*a[1]

# Compute recapitulative tables over all models
recap = table_recap(validation_set, stoploss, takeprofit, nPCs)
recap.to_csv('./Results/Recapitulative_result_stoploss{}_takeprofit{}_{}PCs.csv'.format(stoploss, takeprofit, nPCs))


# And display the most profitable
recap = recap.sort_values('ROI%', ascending = False)
recap = recap[recap['nTrades'] > 50] # let's say we want strategies with at least 50 trades over the validation set
min, max = recap['Min'].iloc[0], recap['Max'].iloc[0]

# Compute predictions on testset
testset['preds'] = (clf.predict(testset_final.iloc[:, :nPCs]) > 0.5)*1
testset['proba1'] = clf.predict(testset_final.iloc[:, :nPCs])

# Compute earnings column
a = compute_earnings_loss(stoploss, takeprofit, fees)
testset['EarningsBullish'] = (testset['preds'] == testset['result'])*a[0] + (testset['preds'] != testset['result'])*a[1]

# keep only the timesteps in which the model predicts a bullish trend
testset2 = testset[testset['preds'] == 1].copy()

# let's say we seek for strategies that display at least n trades over the testing period
testset2 = testset[(testset['proba1'] > min) & (testset['proba1'] < max)].copy()

# Now plot our trading strategy
plt.plot(pd.to_datetime(testset2['date']), np.cumsum(testset2['EarningsBullish']))
plt.title('ROI = {} %'.format(100*np.mean(testset2['EarningsBullish'])))
plt.xlabel('Date')
plt.xlabel('Cumulative Earnings')
plt.show()

# Display the entry points
plt.plot(pd.to_datetime(testset['date']), testset['close'])
plt.scatter(pd.to_datetime(testset2['date']), testset2['close'], c = (testset2['EarningsBullish']>0))
plt.title('Entry points \n Yellow = Win, Blue = Loss')
plt.show()