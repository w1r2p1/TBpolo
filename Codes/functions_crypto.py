'''See https://seb943.github.io/Data/Paper_CreatingATradingBot.pdf for associated report,
some parts were taken from external packages, which are mentioned in the paper'''

import numpy as np
from numba import jit, cuda
import pandas as pd

from math import pi
from time import time
from poloniex import Poloniex
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from datetime import date, datetime
import matplotlib.pyplot as plt
import csv
import pickle as pk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.externals.joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import sys

from trendy import *

def supres(ltp, n):
    """
    This function takes a numpy array of last traded price
    and returns a list of support and resistance levels
    respectively. n is the number of entries to be scanned.
    """
    from scipy.signal import savgol_filter as smooth

    #converting n to a nearest even number
    if n%2 != 0:
        n += 1

    n_ltp = ltp.shape[0]

    # smoothening the curve
    ltp_s = smooth(ltp, (n+1), 3)

    #taking a simple derivative
    ltp_d = np.zeros(n_ltp)
    ltp_d[1:] = np.subtract(ltp_s[1:], ltp_s[:-1])

    resistance = []
    support = []

    for i in xrange(n_ltp - n):
        arr_sl = ltp_d[i:(i+n)]
        first = arr_sl[:(n/2)] #first half
        last = arr_sl[(n/2):] #second half

        r_1 = np.sum(first > 0)
        r_2 = np.sum(last < 0)

        s_1 = np.sum(first < 0)
        s_2 = np.sum(last > 0)

        #local maxima detection
        if (r_1 == (n/2)) and (r_2 == (n/2)):
            resistance.append(ltp[i+((n/2)-1)])

        #local minima detection
        if (s_1 == (n/2)) and (s_2 == (n/2)):
            support.append(ltp[i+((n/2)-1)])

    return support, resistance


def simulate_expectancy(p, stoploss, takeprofit, fees = 0.0009):
    '''hence we need 46% of winning bets with stoploss = 0.5% and takeprofit = 1%)'''
    win = (1-fees)*(1+takeprofit)*(1-fees)
    loss = (1-fees)*(1-stoploss)*(1-fees)
    E = p*win + (1-p)*loss
    ROI = E - 1
    #print('win = ',win,'\n','loss =',loss,'\n','Expectancy = ',E, '\n',' ROI =',100 *ROI, "%")
    return(win, loss, E)


def track_investment(pair, stake, stoploss, takeprofit):
    period = api.MINUTE * 1
    while True :
        raw = api.returnChartData(pair, period = period, start = time() - api.HOUR*100)
        df = pd.DataFrame(raw)
        price = df["close"].iloc[-1]
        if price > takeprofit :
            #order_nb = sell(pair, price, stake, fillOrKill = 1)
            return(order_nb)
        elif price < stoploss :
            #order_nb = sell(pair, price, stake, fillOrKill = 1)
            return(order_nb)


def check_outcome(df, line, stoploss, takeprofit):
    '''
    0 means we reached stoploss
    1 means we reached takeprofit
    -1 means still in between
    '''
    price0 = df["close"].iloc[line]
    upper_lim = price0*(1+takeprofit)
    down_lim = price0*(1-stoploss)
    for i in range(line, df["close"].size):
        if df["low"].iloc[i] < down_lim :
            return(0)
        elif df["high"].iloc[i] > upper_lim :
            return(1)
    return(-1)


def check_outcome_date(df, line, stoploss, takeprofit):
    '''
    0 means we reached stoploss
    1 means we reached takeprofit
    -1 means still in between
    '''
    price0 = df["close"].iloc[line]
    upper_lim = price0*(1+takeprofit)
    down_lim = price0*(1-stoploss)
    for i in range(line, df["close"].size):
        if df["low"].iloc[i] < down_lim :
            return(0, df["date"].iloc[i])
        elif df["high"].iloc[i] > upper_lim :
            return(1, df["date"].iloc[i])
    return(-1, df["date"].iloc[-1])





def compute_variables1(df):
    df['bodysize'] = df['close'] - df['open']
    df['shadowsize'] = df['high'] - df['low']
    df['percentChange'] = df['close'].pct_change()
    for window in [3, 8, 21, 55, 144, 377, 987, 2584, 6765, 10946]:
        print(window)
        df = sma(df, window, targetcol = 'close', colname = 'slow_sma_{}'.format(window))
        df = sma(df, int(window//3), targetcol = 'close', colname = 'fast_sma_{}'.format(window))
        df = bbands(df, window)
        #df = ema(df, window, colname='ema_{}'.format(window))
        df = macd(df, fastcol='fast_sma_{}'.format(window), slowcol='slow_sma_{}'.format(window), colname='macd_{}'.format(window))
        df = rsi(df, window, colname = 'rsi_{}'.format(window))
        df["Min_{}".format(window)] = df["low"].rolling(window).min()
        df["Max_{}".format(window)] = df["high"].rolling(window).max()
        df["volume_{}".format(window)] = df["volume"].rolling(window).mean()
        df["quoteVolume_{}".format(window)] = df["quoteVolume"].rolling(window).mean()
        df["Slope_uptrend_{}".format(window)] = 0
        df["Slope_downtrend_{}".format(window)] = 0
        df["Upslope_diff_{}".format(window)] = 0
        df["Downslope_diff_{}".format(window)] = 0
        df["Upslope_pct_{}".format(window)] = 0
        df["Downslope_pct_{}".format(window)] = 0
        if (window > 3) & (window < 3000):
            for line in range(6*window + 1, len(df["close"])):
                if line % 10000 == 0:
                    print(line, "/", len(df["date"]),'-', window)
                a = segtrends(df["close"][line - 6*window:line], segments=2, charts=False)
                df["Slope_uptrend_{}".format(window)].iloc[line] = (a[1][1] - a[1][0]) / (a[0][1] - a[0][0])
                df["Slope_downtrend_{}".format(window)].iloc[line] = (a[3][1] - a[3][0]) / (a[2][1] - a[2][0])
                df["Upslope_diff_{}".format(window)] = a[4][-1] - df["close"].iloc[line]
                df["Downslope_diff_{}".format(window)] = a[5][-1] - df["close"].iloc[line]
                df["Upslope_pct_{}".format(window)] = (a[4][-1] - df["close"].iloc[line]) / df["close"].iloc[line]
                df["Downslope_pct_{}".format(window)] = (a[5][-1] - df["close"].iloc[line]) / df["close"].iloc[line]
    # (a) Add modulo 10, 100, 1000, 500, 50
    df["Modulo_10"] = df["close"].copy() % 10
    df["Modulo_100"] = df["close"].copy() % 100
    df["Modulo_1000"] = df["close"].copy() % 1000
    df["Modulo_500"] = df["close"].copy() % 500
    df["Modulo_50"] = df["close"].copy() % 50
    # (b) Add weekday and day of the month
    df["WeekDay"] = pd.to_datetime(df["date"]).dt.weekday
    df["Day"] = pd.to_datetime(df["date"]).dt.day
    #df.dropna(inplace=True)
    return(df)


def rsi(df, window, targetcol='weightedAverage', colname='rsi'):
    """ Calculates the Relative Strength Index (RSI) from a pandas dataframe
    http://stackoverflow.com/a/32346692/3389859
    """
    series = df[targetcol]
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
    rs = u.ewm(com=window - 1,
               ignore_na=False,
               min_periods=0,
               adjust=False).mean() / d.ewm(com=window - 1,
                                            ignore_na=False,
                                            min_periods=0,
                                            adjust=False).mean()
    df[colname] = 100 - 100 / (1 + rs)
    df[colname].fillna(df[colname].mean(), inplace=True)
    return df

def sma(df, window, targetcol='close', colname='sma'):
    """ Calculates Simple Moving Average on a 'targetcol' in a pandas dataframe
    """
    df[colname] = df[targetcol].rolling(
        min_periods=1, window=window, center=False).mean()
    return df

def ema(df, window, targetcol='close', colname='ema', **kwargs):
    """ Calculates Expodential Moving Average on a 'targetcol' in a pandas
    dataframe """
    df[colname] = df[targetcol].ewm(
        span=window,
        min_periods=kwargs.get('min_periods', 1),
        adjust=kwargs.get('adjust', True),
        ignore_na=kwargs.get('ignore_na', False)
    ).mean()
    df[colname].fillna(df[colname].mean(), inplace=True)
    return df

def macd(df, fastcol='emafast', slowcol='sma', colname='macd'):
    """ Calculates the difference between 'fastcol' and 'slowcol' in a pandas
    dataframe """
    df[colname] = df[fastcol] - df[slowcol]
    return df

def bbands(df, window, targetcol='close', stddev=2.0):
    """ Calculates Bollinger Bands for 'targetcol' of a pandas dataframe """
    if not 'sma' in df:
        df = sma(df, window, targetcol)
    df['sma'].fillna(df['sma'].mean(), inplace=True)
    df['bbtop'] = df['sma'] + stddev * df[targetcol].rolling(
        min_periods=1,
        window=window,
        center=False).std()
    df['bbtop'].fillna(df['bbtop'].mean(), inplace=True)
    df['bbbottom'] = df['sma'] - stddev * df[targetcol].rolling(
        min_periods=1,
        window=window,
        center=False).std()
    df['bbbottom'].fillna(df['bbbottom'].mean(), inplace=True)
    df['bbrange'] = df['bbtop'] - df['bbbottom']
    df['bbpercent'] = ((df[targetcol] - df['bbbottom']) / df['bbrange']) - 0.5
    return df


def plotRSI(p, df, plotwidth=800, upcolor='green', downcolor='red'):
    # create y axis for rsi
    p.extra_y_ranges = {"rsi": Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name="rsi"), 'right')

    # create rsi 'zone' (30-70)
    p.patch(np.append(df['date'].values, df['date'].values[::-1]),
            np.append([30 for i in df['rsi'].values],
                      [70 for i in df['rsi'].values[::-1]]),
            color='olive',
            fill_alpha=0.2,
            legend="rsi",
            y_range_name="rsi")

    candleWidth = (df.iloc[2]['date'].timestamp() -
                   df.iloc[1]['date'].timestamp()) * plotwidth
    # plot green bars
    inc = df.rsi >= 50
    p.vbar(x=df.date[inc],
           width=candleWidth,
           top=df.rsi[inc],
           bottom=50,
           fill_color=upcolor,
           line_color=upcolor,
           alpha=0.5,
           y_range_name="rsi")
    # Plot red bars
    dec = df.rsi <= 50
    p.vbar(x=df.date[dec],
           width=candleWidth,
           top=50,
           bottom=df.rsi[dec],
           fill_color=downcolor,
           line_color=downcolor,
           alpha=0.5,
           y_range_name="rsi")


def plotMACD(p, df, color='blue'):
    # plot macd
    p.line(df['date'], df['macd'], line_width=4,
           color=color, alpha=0.8, legend="macd")
    p.yaxis[0].formatter = NumeralTickFormatter(format='0.00000000')


def plotCandlesticks(p, df, plotwidth=750, upcolor='green', downcolor='red'):
    candleWidth = (df.iloc[2]['date'].timestamp() -
                   df.iloc[1]['date'].timestamp()) * plotwidth
    # Plot candle 'shadows'/wicks
    p.segment(x0=df.date,
              y0=df.high,
              x1=df.date,
              y1=df.low,
              color="black",
              line_width=2)
    # Plot green candles
    inc = df.close > df.open
    p.vbar(x=df.date[inc],
           width=candleWidth,
           top=df.open[inc],
           bottom=df.close[inc],
           fill_color=upcolor,
           line_width=0.5,
           line_color='black')
    # Plot red candles
    dec = df.open > df.close
    p.vbar(x=df.date[dec],
           width=candleWidth,
           top=df.open[dec],
           bottom=df.close[dec],
           fill_color=downcolor,
           line_width=0.5,
           line_color='black')
    # format price labels
    p.yaxis[0].formatter = NumeralTickFormatter(format='0.00000000')


def plotVolume(p, df, plotwidth=800, upcolor='green', downcolor='red'):
    candleWidth = (df.iloc[2]['date'].timestamp() -
                   df.iloc[1]['date'].timestamp()) * plotwidth
    # create new y axis for volume
    p.extra_y_ranges = {"volume": Range1d(start=min(df['volume'].values),
                                          end=max(df['volume'].values))}
    p.add_layout(LinearAxis(y_range_name="volume"), 'right')
    # Plot green candles
    inc = df.close > df.open
    p.vbar(x=df.date[inc],
           width=candleWidth,
           top=df.volume[inc],
           bottom=0,
           alpha=0.1,
           fill_color=upcolor,
           line_color=upcolor,
           y_range_name="volume")

    # Plot red candles
    dec = df.open > df.close
    p.vbar(x=df.date[dec],
           width=candleWidth,
           top=df.volume[dec],
           bottom=0,
           alpha=0.1,
           fill_color=downcolor,
           line_color=downcolor,
           y_range_name="volume")


def plotBBands(p, df, color='navy'):
    # Plot bbands
    p.patch(np.append(df['date'].values, df['date'].values[::-1]),
            np.append(df['bbbottom'].values, df['bbtop'].values[::-1]),
            color=color,
            fill_alpha=0.1,
            legend="bband")
    # plot sma
    p.line(df['date'], df['sma'], color=color, alpha=0.9, legend="sma")


def plotMovingAverages(p, df):
    # Plot moving averages
    p.line(df['date'], df['emaslow'],
           color='orange', alpha=0.9, legend="emaslow")
    p.line(df['date'], df['emafast'],
           color='red', alpha=0.9, legend="emafast")


class Charter(object): # maybe change object to api
    """ Retrieves 5min candlestick data for a market and saves it in a mongo
    db collection. Can display data in a dataframe or bokeh plot."""

    def __init__(self, api):
        """
        api = poloniex api object
        """
        self.api = api

    def __call__(self, pair, frame=False):
        """ returns raw chart data from the mongo database, updates/fills the
        data if needed, the date column is the '_id' of each candle entry, and
        the date column has been removed. Use 'frame' to restrict the amount
        of data returned.
        Example: 'frame=api.YEAR' will return last years data
        """
        # use last pair and period if not specified
        if not frame:
            frame = self.api.YEAR * 10
        dbcolName = pair + 'chart'
        # get db connection
        db = MongoClient()['poloniex'][dbcolName]
        # get last candle
        try:
            last = sorted(
                list(db.find({"_id": {"$gt": time() - 60 * 20}})),
                key=itemgetter('_id'))[-1]
        except:
            last = False
        # no entrys found, get all 5min data from poloniex
        if not last:
            logger.warning('%s collection is empty!', dbcolName)
            new = self.api.returnChartData(pair,
                                           period=60 * 5,
                                           start=time() - self.api.WEEK * 13)
        else:
            new = self.api.returnChartData(pair,
                                           period=60 * 5,
                                           start=int(last['_id']))
        # add new candles
        updateSize = len(new)
        logger.info('Updating %s with %s new entrys!',
                    dbcolName, str(updateSize))

        # show the progess
        for i in range(updateSize):
            print("\r%s/%s" % (str(i + 1), str(updateSize)), end=" complete ")
            date = new[i]['date']
            del new[i]['date']
            db.update_one({'_id': date}, {"$set": new[i]}, upsert=True)
        print('')

        logger.debug('Getting chart data from db')
        # return data from db (sorted just in case...)
        return sorted(
            list(db.find({"_id": {"$gt": time() - frame}})),
            key=itemgetter('_id'))

    def dataFrame(self, pair, frame=False, zoom=False, window=120):
        """ returns pandas DataFrame from raw db data with indicators.
        zoom = passed as the resample(rule) argument to 'merge' candles into a
            different timeframe
        window = number of candles to use when calculating indicators
        """
        data = self.__call__(pair, frame)
        # make dataframe
        df = pd.DataFrame(data)
        # set date column
        df['date'] = pd.to_datetime(df["_id"], unit='s')
        if zoom:
            df.set_index('date', inplace=True)
            df = df.resample(rule=zoom,
                             closed='left',
                             label='left').apply({'open': 'first',
                                                  'high': 'max',
                                                  'low': 'min',
                                                  'close': 'last',
                                                  'quoteVolume': 'sum',
                                                  'volume': 'sum',
                                                  'weightedAverage': 'mean'})
            df.reset_index(inplace=True)

        # calculate/add sma and bbands
        df = bbands(df, window)
        # add slow ema
        df = ema(df, window, colname='emaslow')
        # add fast ema
        df = ema(df, int(window // 3.5), colname='emafast')
        # add macd
        df = macd(df)
        # add rsi
        df = rsi(df, window // 5)
        # add candle body and shadow size
        df['bodysize'] = df['close'] - df['open']
        df['shadowsize'] = df['high'] - df['low']
        df['percentChange'] = df['close'].pct_change()
        df.dropna(inplace=True)
        return df

    def graph(self, pair, frame=False, zoom=False,
              window=120, plot_width=1000, min_y_border=40,
              border_color="whitesmoke", background_color="white",
              background_alpha=0.4, legend_location="top_left",
              tools="pan,wheel_zoom,reset"):
        """
        Plots market data using bokeh and returns a 2D array for gridplot
        """
        df = self.dataFrame(pair, frame, zoom, window)
        #
        # Start Candlestick Plot -------------------------------------------
        # create figure
        candlePlot = figure(
            x_axis_type=None,
            y_range=(min(df['low'].values) - (min(df['low'].values) * 0.2),
                     max(df['high'].values) * 1.2),
            x_range=(df.tail(int(len(df) // 10)).date.min().timestamp() * 1000,
                     df.date.max().timestamp() * 1000),
            tools=tools,
            title=pair,
            plot_width=plot_width,
            plot_height=int(plot_width // 2.7),
            toolbar_location="above")
        # add plots
        # plot volume
        plotVolume(candlePlot, df)
        # plot candlesticks
        plotCandlesticks(candlePlot, df)
        # plot bbands
        plotBBands(candlePlot, df)
        # plot moving aves
        plotMovingAverages(candlePlot, df)
        # set legend location
        candlePlot.legend.location = legend_location
        # set background color
        candlePlot.background_fill_color = background_color
        candlePlot.background_fill_alpha = background_alpha
        # set border color and size
        candlePlot.border_fill_color = border_color
        candlePlot.min_border_left = min_y_border
        candlePlot.min_border_right = candlePlot.min_border_left
        #
        # Start RSI/MACD Plot -------------------------------------------
        # create a new plot and share x range with candlestick plot
        rsiPlot = figure(plot_height=int(candlePlot.plot_height // 2.5),
                         x_axis_type="datetime",
                         y_range=(-(max(df['macd'].values) * 2),
                                  max(df['macd'].values) * 2),
                         x_range=candlePlot.x_range,
                         plot_width=candlePlot.plot_width,
                         title=None,
                         toolbar_location=None)
        # plot macd
        plotMACD(rsiPlot, df)
        # plot rsi
        plotRSI(rsiPlot, df)
        # set background color
        rsiPlot.background_fill_color = candlePlot.background_fill_color
        rsiPlot.background_fill_alpha = candlePlot.background_fill_alpha
        # set border color and size
        rsiPlot.border_fill_color = candlePlot.border_fill_color
        rsiPlot.min_border_left = candlePlot.min_border_left
        rsiPlot.min_border_right = candlePlot.min_border_right
        rsiPlot.min_border_bottom = 20
        # orient x labels
        rsiPlot.xaxis.major_label_orientation = pi / 4
        # set legend
        rsiPlot.legend.location = legend_location
        # set dataframe 'date' as index
        df.set_index('date', inplace=True)
        # return layout and df
        return [[candlePlot], [rsiPlot]], df



def create_recap_test_table(stoploss, takeprofit, nPCs, pair, period, fees = 0.0009):
    ''' Creates a table which contains columns Result, duration, Earnings0, Earnings1, estimated percentages'''
    import os
    os.chdir("C:\\Users\\Sébastien CARARO\\Desktop\\Crypto\\Data")
    # (a) Load test table containing variables after PCA
    print(1)
    testset = pd.read_csv("./TestSets/TestSet_{}_{}-2020-03-24-{}-{}.csv".format(pair,period,stoploss,takeprofit))
    # (b) Load model
    print(1)
    with open("./Models/ML/FeedforwardNN3_{}_{}_{}_{}_{}PC.pkl".format(pair,period, stoploss, takeprofit, nPCs), 'rb') as f:
        clf = pk.load(f)
    # (c) Predict percentages
    print(1)
    new_pca_df = testset.iloc[:, :nPCs]
    preds = (clf.predict(new_pca_df) > 0.5)*1
    probas = pd.DataFrame(clf.predict_proba(new_pca_df))

    proba0 = list(1-probas.iloc[:,0])
    proba1 =  list(probas.iloc[:,0])

    # (d) Create Earnings0 and Earnings1 columns
    print(1)
    win1 = simulate_expectancy(1, stoploss, takeprofit, fees = 0.0009)[0]
    loss1 = simulate_expectancy(1, stoploss, takeprofit, fees = 0.0009)[1]
    testset['Earnings0'], testset['Earnings1'] = 0, 0

    for i in range(len(testset['Earnings1'])):
        if testset["result"].iloc[i] == 1:
            testset['Earnings1'].iloc[i] = win1 - 1
            testset['Earnings0'].iloc[i] = -(win1 - 1)
        elif testset["result"].iloc[i] == 0:
            testset['Earnings1'].iloc[i] = loss1 - 1
            testset['Earnings0'].iloc[i] = -(loss1 - 1)

    #testset['EarningsPred'] = (preds == 0 * testset["Earnings0"]) + (preds == 1 * testset["Earnings1"])

    # (e) Bind all information in one single table
    print(1)
    table = testset[['date', 'end', 'result', 'Earnings0', 'Earnings1']]
    table['pred'] = preds
    table['proba0'] = proba0
    table['proba1'] = proba1

    return(table)



def table_recap(stoploss, takeprofit, nPCs, columnA = 'proba1', columnB = 'EarningsPred1', strategy = 'Bullish'):
    ''' Summarize the strategy by steps of 0.05, depending on which column (i.e. strategy)
    we choose'''
    df = pd.read_csv('./Results/{}/{}Results_stoploss{}_takeprofit{}_nPCs{}.csv'.format(strategy, strategy, stoploss, takeprofit, nPCs))
    subdivisions = 10 # number of subdivisions of [0.5, 1] that we want (arbitrary)
    m = subdivisions*(subdivisions+1) / 2

    recap = pd.DataFrame(np.zeros((int(m), 0)))
    recap['stoploss'] = stoploss
    recap['takeprofit'] = takeprofit
    recap['nPCs'] = nPCs
    recap['strategy'] = strategy
    recap['Min'] = 0
    recap['Max'] = 0
    c= 0
    l = list(np.arange(0.5, 1.05, (1-0.5)/subdivisions))
    for i in range(subdivisions):
        for j in range(i + 1, subdivisions + 1):
            #print(l[i],l[j])
            recap['Min'].iloc[c] = l[i]
            recap['Max'].iloc[c] = l[j]
            c+=1
    recap['ROI%'] = 0
    #recap['ROI%.min-1'] = 0
    recap['nTrades'] = 0
    for i in range(len(recap['Min'])):
        min, max = recap['Min'].iloc[i], recap['Max'].iloc[i]
        #print(min, max)
        df2 = df[(df[columnA] > min) & (df[columnA] < max)]
        recap['ROI%'].iloc[i] = 100 * np.mean(df2[columnB])
        #recap['ROI%.min-1'].iloc[i] = np.mean(100 * 60 * df2[columnB] / pd.to_timedelta(df2['duration']).dt.total_seconds())
        recap['nTrades'].iloc[i] = df2.shape[0]

    recap.to_csv('./Results/{}/Recap{}_stoploss{}_takeprofit{}_nPCs{}.csv'.format(strategy, strategy, stoploss, takeprofit, nPCs))

    return(recap)