# IV. Deploy the trading bot in Poloniex for One trade at a time
from poloniex import Poloniex
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import poloniex
import os
from time import time, sleep
import pickle as pk
import re

from Deployment_functions import *

os.chdir('C:/Users/Utilisateur/Desktop/Crypto/Data')

######################### 0 - Load necessary data, define constants
# (0) Define constants used for the trading loop
pair = 'USDT_BTC'
period = 7200 # In our example, we are building a 2 hours (7200 secs) trading bot
stoploss = 0.05
takeprofit = 0.1

buy_signal = 0
sell_signal = 0

amount_dollar = 50 # amount in dollar that we invest for each trade

# (i) Load scalers and PCA
with open('./Models/scaler.pkl', 'rb') as f:
    scale_fct = pk.load(f)
with open('./Models/pca.pkl', 'rb') as f:
    pca = pk.load(f)
with open('./Models/pca_scaler.pkl', 'rb') as f:
    pca_scaler = pk.load(f)

# (ii) Pick the best model
recap = pd.read_csv('./Results/Comparative_All_models_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit)).sort_values('Accuracy', ascending = False)
nPCs = recap['nPCs'].iloc[0]

with open("./Models/DL_model_{}PC_stoploss{}_takeprofit{}.pkl".format(nPCs, stoploss, takeprofit), 'rb') as f:
    clf = pk.load(f)

# (iii) According to your strategy (see article III), define a min and a max thresholds for following it (by default : 0.5 and 1)
min_threshold, max_threshold = 0.5, 1
######################### I - Initialize Poloniex API connection
api_key = getkeys()[0]
api_secret = getkeys()[1]

polo = Poloniex(api_key, api_secret)
######################### II - One trade at a time trading bot
try:
    while True:
        if int(time())%period == 0 :
            print('It is {}'.format(datetime.now()), "... Let's trade {} ! ".format(pair))
            # I- Request the data
            raw = polo.returnChartData(pair, period = period, start = int(time()) - period*1000)
            df = pd.DataFrame(raw).iloc[1:] # First row may contain useless data
            while (time() - df['date'].iloc[-1]) > period : #Check we've got the very recent candle : we stay here until Poloniex delivers it
                print('Waiting for actualized data...');sleep(1)
                raw = polo.returnChartData(pair, period = period, start = int(time()) - period*1000)
                df = pd.DataFrame(raw).iloc[1:] # First row may contain useless data
            df['date'] = pd.to_datetime(df["date"], unit='s')
            df = df[['close', 'date', 'high', 'low', 'open', 'volume']] # only keep required data


            # II - Compute predictions
            df = compute_variables1(df)
            df_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(df.drop('date', 1)))))
            df_final = df_final.iloc[:, :nPCs]

            df['preds'] = (clf.predict(df_final.iloc[:, :nPCs]) > 0.5)*1
            df['proba1'] = clf.predict(df_final.iloc[:, :nPCs])

            # III - If criterion reached, we buy the asset and track the trade until we reach either the takeprofit or the stoploss
            buy_signal = ((df['proba1'].iloc[-1] > min_threshold) & (df['proba1'].iloc[-1] < max_threshold))*1
            if buy_signal:
                recap_trade1 = buy_asset(pair = pair, amount = amount_dollar, store = True)
                price = np.mean(recap_trade1['Rate'])
                amount = np.sum(recap_trade1['AmountInCurrency'])
                recap_trade2 = track_investment(pair, price, amount, stoploss, takeprofit)
                print('We made a profit of {} $ with this trade!'.format(np.sum(recap_trade1['Total$']) - np.sum(recap_trade2['Total$Adjusted'])))
except:
    # If the loop encountered an error, just sell everything
    currency = re.split('_', pair)[1]
    sell_everything(currency)
