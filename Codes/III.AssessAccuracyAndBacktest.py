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

from scipy.stats import ttest_1samp

################### (a) Load previously built datasets
stoploss = 0.05
takeprofit = 0.1
fees = 0.00125 # transaction fees : 0.125% for example

nTrades_mini = 50 # minimal number of trades we want over the test set: this is for second approach

list_nPCs = [10, 15, 20, 25, 30, 35, 40]

trainset_final = pd.read_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
trainset = pd.read_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

validation_set_final = pd.read_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
validation_set = pd.read_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))

testset_final = pd.read_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))
testset = pd.read_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit))


def compute_earnings_loss(stoploss, takeprofit, fees):
    '''Compute earnings and loss with given fees, stoploss, takeprofit'''
    win = (1-fees)*(1+takeprofit)*(1-fees) -1
    loss = (1-fees)*(1-stoploss)*(1-fees) -1
    return(win, loss)

def predict_and_backtest_bullish(df, df_final, model, stoploss, takeprofit, fees, plotting = True):
    '''This functin takes the test set as input (in both shapes) + the model, computes predictions and  probabilities, then compute the earnings according to the fees. Finally it can plot the strategy'''
    # Compute predictions on testset
    df['preds'] = (clf.predict(df_final.iloc[:, :nPCs]) > 0.5)*1
    df['proba1'] = clf.predict(df_final.iloc[:, :nPCs])

    # keep only the timesteps in which the model predicts a bullish trend
    testset1 = df[df['preds'] == 1].copy()

    # Compute earnings column
    a = compute_earnings_loss(stoploss, takeprofit, fees)
    testset1['EarningsBullish'] = (testset1['preds'] == testset1['result'])*a[0] + (testset1['preds'] != testset1['result'])*a[1]

    if plotting:
        # Now plot our trading strategy
        plt.plot(pd.to_datetime(testset1['date']), np.cumsum(testset1['EarningsBullish']))
        plt.title('Best model over the testset \n ROI = {} %'.format(100*np.mean(testset1['EarningsBullish'])))
        plt.xlabel('Date')
        plt.xlabel('Cumulative Earnings')
        plt.show()

        # Display the entry points
        plt.plot(pd.to_datetime(df['date']), df['close'])
        plt.scatter(pd.to_datetime(testset1['date']), testset1['close'], c = (testset1['EarningsBullish']>0))
        plt.title('Entry points \n Yellow = Win, Blue = Loss')
        plt.show()

    return(testset1)

#################### (b) Basic strategy : pick the best model and bet on bullish trends over the testset
recap = pd.read_csv('Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]

with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    clf = pk.load(f)

testset1 = predict_and_backtest_bullish(testset, testset_final, clf, stoploss, takeprofit, fees, plotting = True)

# Assess the performance by comparing to if we always traded bullish blindly over the period
a = compute_earnings_loss(stoploss, takeprofit, fees)
testset_benchmark = testset.copy()
testset_benchmark['EarningsBullish'] = (testset['result'] == 1)*a[0] + (testset['result'] == 0)*a[1]
avg_return_benchmark = np.mean(testset_benchmark['EarningsBullish'])

# Now let's look at our approach's performance and std
p_value = ttest_1samp(testset1['EarningsBullish'], popmean = avg_return_benchmark)[1]
print('Our model has an average ROI of {} %, while trading blindly bullish over the same period yielded a ROI of {} %, when we perform statistical testing of difference there is a p-value of {}.'.format(100*np.mean(testset1['EarningsBullish']), 100*avg_return_benchmark, p_value))

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

# (i) Identify best threshold
# Load best model
recap = pd.read_csv('Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]
with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    clf = pk.load(f)

# Compute predictions on validation_set
validation_set['preds'] = (clf.predict(validation_set_final.iloc[:, :nPCs]) > 0.5)*1
validation_set['proba1'] = clf.predict(validation_set_final.iloc[:, :nPCs])
a = compute_earnings_loss(stoploss, takeprofit, fees)
validation_set['EarningsBullish'] = (validation_set['preds'] == validation_set['result'])*a[0] + (validation_set['preds'] != validation_set['result'])*a[1]

# Compute recapitulative table
recap = table_recap(validation_set, stoploss, takeprofit, nPCs)
recap.to_csv('./Results/Recapitulative_result_stoploss{}_takeprofit{}_{}PCs.csv'.format(stoploss, takeprofit, nPCs), index = False)

# Pick the most profitable
recap = recap.sort_values('ROI%', ascending = False)
recap = recap[recap['nTrades'] > nTrades_mini]
print(recap)
min, max = recap['Min'].iloc[0], recap['Max'].iloc[0]


# (ii) Now that we have identified the best threshold, filter the predictions
# ...Finally : we plot our strategy
testset2 = predict_and_backtest_bullish(testset, testset_final, clf, stoploss, takeprofit, fees, plotting = False)
testset2 = testset2[(testset2['proba1'] > min) & (testset2['proba1'] < max)].copy()

# Now plot our trading strategy
plt.plot(pd.to_datetime(testset2['date']), np.cumsum(testset2['EarningsBullish']))
plt.title('Best model over the testset \n ROI = {} %'.format(100*np.mean(testset2['EarningsBullish'])))
plt.xlabel('Date')
plt.xlabel('Cumulative Earnings')
plt.show()

# Display the entry points
plt.plot(pd.to_datetime(testset['date']), testset['close'])
plt.scatter(pd.to_datetime(testset2['date']), testset2['close'], c = (testset2['EarningsBullish']>0))
plt.title('Entry points \n Yellow = Win, Blue = Loss')
plt.show()

# (iii) Assess the performance by comparing to if we always traded bullish over the period
a = compute_earnings_loss(stoploss, takeprofit, fees)
testset_benchmark = testset.copy()
testset_benchmark['EarningsBullish'] = (testset['result'] == 1)*a[0] + (testset['result'] == 0)*a[1]
avg_return_benchmark = np.mean(testset_benchmark['EarningsBullish'])

# Now let's look at our approach's performance and std
p_value = ttest_1samp(testset2['EarningsBullish'], popmean = avg_return_benchmark)[1]
print('Our model has an average ROI of {} %, while trading blindly bullish over the same period yielded a ROI of {} %, when we perform statistical testing of difference there is a p-value of {}.'.format(100*np.mean(testset2['EarningsBullish']), 100*avg_return_benchmark, p_value))