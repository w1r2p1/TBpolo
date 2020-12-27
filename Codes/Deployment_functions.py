# Deployment functions


############################# Functions realted to variables computation
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
        df['percentChange_{}'.format(window)] = df['close'].pct_change(periods = window)
        df['RelativeSize_sma_{}'.format(window)] = df['close'] / df['sma_{}'.format(window)]
        df['Diff_{}'.format(window)] = df['close'].diff(window)

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




############ Then the functions related to the Poloniex API
def getkeys():
    # Fill in you own api key & secret
    api_key = ''
    api_secret = ''
    return(api_key, api_secret)

def buy_asset(pair, amount, store = True):
    '''Given a pair and an amount in $, buy the amount at the lowest possible price'''
    total_bought = 0
    total_to_buy = amount
    res = []
    total = 0
    # I - Sell
    while total_bought < 0.95*amount: # Let's say 95% is acceptable : in the vast majority of cases we will achieve 100% with the first trade, though...
        # Order to buy the missing amount we want to sell, then update the total amount bought
        rate = polo.returnTicker()[pair]["lowestAsk"]
        amount_pair = total_to_buy/rate
        res2 = polo.buy(currencyPair=pair, rate= rate, amount = amount_pair, immediateOrCancel = 1)
        trade = res2['resultingTrades']
        total = [trade[k]['total'] for k in range(len(trade))]
        total_bought = total_bought + np.sum(total)
        total_to_buy = total_to_buy - np.sum(total)
        # Update the recapitulative of trades passed
        res = res + [res2]

    # II - Create a recapitulative table
    trades = [res[k]['resultingTrades'] for k in range(len(res))]

    dates = [trades[i][0]['date'] for i in range(len(trades))]
    amounts = [trades[i][0]['takerAdjustment'] for i in range(len(trades))] # amount bought expressed in the pair
    totals = [trades[i][0]['total'] for i in range(len(trades))] #amounts expressed in dollar
    rates = [trades[i][0]['rate'] for i in range(len(trades))] #rate of the pair for each trade

    fees = res[0]['fee']

    recap= pd.DataFrame({'Date' : dates, 'AmountInCurrency' : amounts, 'Total$' : totals, 'Rate' : rates})
    recap['FeesPaid$'] = recap['Total$']*fees
    recap['Pair'] = pair
    recap['Type'] = 'Buy'

    # III - If selected, store the trades in a csv
    if store:
        if not os.path.exists('./Trades'):
            os.makedirs('./Trades')
        recap.to_csv('./Trades/Buy_{}_{}_{}.csv'.format(pair, amount, res[0]['orderNumber']), index = False, sep = ';')

    print('We bought {} $ of {}. '.format(total_bought, pair))
    return(recap)

def sell_asset(pair, amount, store = True):
    '''Given a pair and an amount expressed in the pair, sell the amount at the highest possible price'''
    amount_sold = 0
    amount_to_sell = amount
    res = []
    amount1 = 0
    # I - Sell
    while amount_sold < 0.95*amount: # Let's say 95% is acceptable : in the vast majority of cases we will achieve 100% with the first trade, though...
        # Order to sell the missing amount we want to sell, then update the total amount sold
        res2 = polo.sell(currencyPair= pair, rate=polo.returnTicker()[pair]["highestBid"], amount=amount_to_sell, immediateOrCancel = 1)
        trade = res2['resultingTrades']
        amount1 = [trade[k]['amount'] for k in range(len(trade))]
        amount_sold = amount_sold + np.sum(amount1)
        amount_to_sell = amount_to_sell - np.sum(amount1)
        # Update the recapitulative of trades passed
        res = res + [res2]

    # II - Create a recapitulative table
    trades = [res[k]['resultingTrades'] for k in range(len(res))]

    dates = [trades[i][0]['date'] for i in range(len(trades))]
    amounts = [trades[i][0]['amount'] for i in range(len(trades))] # amounts expressed in the pair
    totals = [trades[i][0]['total'] for i in range(len(trades))] #amounts expressed in dollar
    totals_adjusted = [trades[i][0]['takerAdjustment'] for i in range(len(trades))] # amounts minus fees expressed in dollar
    rates = [trades[i][0]['rate'] for i in range(len(trades))] #rate of the pair for each trade

    recap= pd.DataFrame({'Date' : dates, 'AmountInCurrency' : amounts, 'Total$' : totals,'Total$Adjusted': totals_adjusted, 'Rate' : rates})
    recap['FeesPaid$'] = recap['Total$'] - recap['Total$Adjusted']
    recap['Pair'] = pair
    recap['Type'] = 'Sell'

    # III - If selected, store the trades in a csv
    if store:
        if not os.path.exists('./Trades'):
            os.makedirs('./Trades')
        recap.to_csv('./Trades/Sell_{}_{}_{}.csv'.format(pair, amount, res[0]['orderNumber']), index = False, sep = ';')

    print('We sold {} of {}. '.format(amount_sold, pair))
    return(recap)


def track_investment(pair, price, amount, stoploss, takeprofit):
    print(stoploss, takeprofit)
    stoploss_price = price*(1-stoploss)
    takeprofit_price = price*(1+takeprofit)
    print('stoploss_price = ', stoploss_price, 'takeprofit_price = ', takeprofit_price)
    while True:
        if polo.returnTicker()[pair]["lowestAsk"] < stoploss_price :
            return(sell_asset(pair, amount, store = True)) # Sell if we reached the stoploss
        if polo.returnTicker()[pair]["highestBid"] > takeprofit_price :
            return(sell_asset(pair, amount, store = True)) # Sell if we reached the takeprofit
        print('{} Latest price is'.format(datetime.now()), polo.returnTicker()[pair]['last'], 'stoploss_price = ', stoploss_price, 'takeprofit_price = ', takeprofit_price)

