# Article I :  Compute predictive variables for financial forecasting
import os
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data, define test set and train set
os.chdir("C:\\Users\\Utilisateur\\Desktop\\Crypto\\Data")
df = pd.read_csv('./Data/USDT_BTC_Poloniex_20022015_21122020_7200.csv')
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(by = 'date')

# Keep only 5 basic information + the date
df = df[['close', 'date', 'high', 'low', 'open', 'volume']]

# Define limit between train set and testset
start_validation = '2018-12-21 12:00:00'
start_test = '2019-12-21 12:00:00'
stoploss = 0.05
takeprofit = 0.1

# Creating subfolders if they don't exist ...
if not os.path.exists('./Data'):
        os.makedirs('./Data')
if not os.path.exists('./Models'):
        os.makedirs('./Models')
if not os.path.exists('./Results'):
        os.makedirs('./Results')

#################### I - Define standard functions ###############################
def compute_sma(df, window, colname):
    '''Computes Simple Moving Average column on a dataframe'''
    df[colname] = df['close'].rolling(window=window, center=False).mean()
    return(df)

def compute_rsi(df, window, colname):
    '''Computes RSI column for a dataframe. http://stackoverflow.com/a/32346692/3389859'''
    series = df['close']
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[window - 1]] = np.mean(u[:window])
    u = u.drop(u.index[:(window - 1)])
    # first value is sum of avg losses
    d[d.index[window - 1]] = np.mean(d[:window])
    d = d.drop(d.index[:(window - 1)])
    rs = u.ewm(com=window - 1,ignore_na=False,
               min_periods=0,adjust=False).mean() / d.ewm(com=window - 1, ignore_na=False,
                                            min_periods=0,adjust=False).mean()
    df[colname] = 100 - 100 / (1 + rs)
    df[colname].fillna(df[colname].mean(), inplace=True)
    return(df)

#################### II - Build the function to compute variables #################
def compute_variables1(df):
    print("Let's compute predictive variables : ")
    df["date"] = pd.to_datetime(df["date"])
    df['bodysize'] = df['close'] - df['open']
    df['shadowsize'] = df['high'] - df['low']
    for window in [3, 8, 21, 55, 144, 377]: # several Fibonacci numbers
        print(window)
        df = compute_sma(df, window, colname = 'sma_{}'.format(window))
        df = compute_rsi(df, window, colname = 'rsi_{}'.format(window))
        df["Min_{}".format(window)] = df["low"].rolling(window).min()
        df["Max_{}".format(window)] = df["high"].rolling(window).max()
        df["volume_{}".format(window)] = df["volume"].rolling(window).mean()
        df['percentChange_{}'.format(window)] = df['close'].pct_change(window = window)
        df['RelativeSize_sma_{}'.format(window)] = df['close'] / df['sma_{}'.format(window)]
    # (a) Add modulo 10, 100, 1000, 500, 50
    df["Modulo_10"] = df["close"].copy() % 10
    df["Modulo_100"] = df["close"].copy() % 100
    df["Modulo_1000"] = df["close"].copy() % 1000
    df["Modulo_500"] = df["close"].copy() % 500
    df["Modulo_50"] = df["close"].copy() % 50
    # (b) Add weekday and day of the month
    df["WeekDay"] = df["date"].dt.weekday
    df["Day"] = df["date"].dt.day
    df.dropna(inplace=True)
    return(df)

df = compute_variables1(df)
df.to_csv('./Data/DatasetWithVariables.csv', index = False)

#################### III - Compute the output #######################################
def check_outcome(df, line, stoploss, takeprofit):
    '''0 means we reached stoploss
    1 means we reached takeprofit
    -1 means still in between'''
    price0 = df["close"].iloc[line]
    upper_lim = price0*(1+takeprofit)
    down_lim = price0*(1-stoploss)
    for i in range(line, df["close"].size):
        if df["low"].iloc[i] < down_lim :
            return(0)
        elif df["high"].iloc[i] > upper_lim :
            return(1)
    return(-1)

def compute_result(df, stoploss, takeprofit):
    df['result'] = 0
    for i in range(df["close"].size):
        if i%500 == 0:
            print(i, '/', df.shape[0])
        df['result'].iloc[i] = check_outcome(df, i, stoploss, takeprofit)
    return(df)

df = compute_result(df, stoploss, takeprofit)
df = df[df['result']>=0] # Only keep observations where we also have the result
df.to_csv('./Data/DatasetWithVariablesAndY_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)

#################### IV - Apply PCA and save results ##############################
# First we define the trainset, validation set, testset. This is important in this step to avoid causality issues.
trainset = df[df['date'] < start_validation]
validation_set = df[(df['date'] >= start_validation) & (df['date'] < start_test)]
testset = df[df['date'] > start_test]

trainset.to_csv('./Data/TrainSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
validation_set.to_csv('./Data/ValidationSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
testset.to_csv('./Data/TestSet_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)

# Display the splitting
plt.plot(pd.to_datetime(trainset['date']), trainset['close'], c = 'orange')
plt.plot(pd.to_datetime(validation_set['date']), validation_set['close'], c = 'b')
plt.plot(pd.to_datetime(testset['date']), testset['close'], c = 'g')
plt.title('Repartition between trainset, validation set, and test set')
plt.show()

# (i) Scale the variables
scale_fct = StandardScaler()
scale_fct.fit(trainset.drop('date', 1).drop('result', 1))
pk.dump(scale_fct, open('./Models/scaler.pkl','wb'))

# (ii) Apply PCA
pca = PCA(n_components=trainset.shape[1] - 2) # remove the result and the date
pca.fit(scale_fct.transform(trainset.drop('date', 1).drop('result', 1)))
pk.dump(pca, open('./Models/pca.pkl',"wb"))

# (iii) Scale PCA components (this accelerates training process in Deep Learning)
pca_scaler = StandardScaler()
pca_scaler.fit(pca.transform(scale_fct.transform(trainset.drop('date', 1).drop('result', 1))))
pk.dump(pca_scaler, open('./Models/pca_scaler.pkl','wb'))

# (iv) Save ready-to-use versions (i.e. datasets after applying scalers and PCA)
trainset_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(trainset.drop('date', 1).drop('result', 1)))))
validation_set_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(validation_set.drop('date', 1).drop('result', 1)))))
testset_final = pd.DataFrame(pca_scaler.transform(pca.transform(scale_fct.transform(testset.drop('date', 1).drop('result', 1)))))

trainset_final.to_csv('./Data/TrainSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
validation_set_final.to_csv('./Data/ValidationSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
testset_final.to_csv('./Data/TestSet_final_stoploss{}_takeprofit{}.csv'.format(stoploss, takeprofit), index = False)
