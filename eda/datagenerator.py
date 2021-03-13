# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

from pandas_datareader import data as pdr
import pandas_ta as ta #commented out for me for now, I (Ben) had some dependency issues

import yfinance as yf
# import quandl as qd # not used, ignor ples
from backtesting import Backtest, Strategy


# %%
yf.pdr_override() # for use with pandas-datareader, optional

# %% [markdown]
# uncomment below if you need to update/generate your monthly and daily csv files.

# %%
# ticker = pd.read_csv('ticker.csv')['Ticker']
# tickers = ticker.to_list() # This is a list of all tickers in the SP500
# tickers = [x.replace('.','-') for x in tickers] # yahoo has '-' instead of '.' for tickers, eg BRK.B

# ## below is how I got monthly and daily pandas dataframes of all stocks in one huge dict.  
# monthly_sp500 = {}
# daily_sp500 = {}
# for tkr in tickers: # run all 500 at your own risk, it takes a while lol
# # for tkr in tickers[:10]:
#     print('Grabbing ' + tkr + " data!")
#     monthly_sp500[tkr] = pdr.get_data_yahoo(tkr, start="2010-01-01", interval = "1mo")
#     daily_sp500[tkr] = pdr.get_data_yahoo(tkr,start="2010-01-01")
#     sleep(.1) # not planning on ddos-ing yahoo today
# monthly_sp500['AAPL'] # take AAPL, for example

# ## everything in one stupid large dataframe
# sp500m = monthly_sp500[tickers[0]] # monthly
# sp500m['Name'] = tickers[0]

# sp500d = daily_sp500[tickers[0]] # daily
# sp500d['Name'] = tickers[0]

# # for tkr in tickers[1:10]:
# for tkr in tickers[1:]:
#     df1 = monthly_sp500[tkr] # monthly
#     df1['Name'] = tkr
#     sp500m = sp500m.append(df1)

#     df2 = daily_sp500[tkr] # daily
#     df2['Name'] = tkr
#     sp500d = sp500d.append(df2)

# sp500m.to_csv('SP500_monthly.csv') # run/uncomment some of these to save these to csv
# sp500d.to_csv('SP500_daily.csv')


# %%
# df = pd.read_csv('SP500_monthly.csv').infer_objects().dropna()
df = pd.read_csv('SP500_daily.csv').infer_objects().dropna()
# df = sp500m.infer_objects().dropna()#.reset_index() # dataframe of all we basically want, OHLC data w/ adjusted close. didnt set any ind
df


# %%
# j=1
# k=2
# # the 'midpoints' for each observe and hold period. midpoint in this context is where we switch from the observation period len=j to the holding period len=k
# j_end = pd.date_range(st+pd.DateOffset(months=j*3), et, freq=str(k*3)+"MS")
# j_end.strftime("%Y-%m-%d").to_list()[:10]


# %%
# pd.date_range(st+pd.DateOffset(days=j), et)


# %%
# def generate_momentum_returns(j=25,k=50):
def generate_daily_timetable(j=10, k=20):
    '''aims to generate the MONTHLY table of times for each observe/hold period. The midpoint in this context is where we switch from the observation period len=j to the holding period len=k'''
    j_end = pd.date_range(pd.to_datetime(df.Date.min())+pd.DateOffset(days=j), pd.to_datetime(df.Date.max()), freq=str(k)+"D") # the 'midpoints' for each observe and hold period. 
    j_start = j_end + pd.DateOffset(days=-j) # based on the midpoint, get the start point for each observe and hold period
    k_end = j_end + pd.DateOffset(days=k-1) # based on the midpoint, get the end point for each observe and hold period
    timetable = pd.DataFrame({'j_start': j_start, 'midpoint': j_end, 'k_end': k_end}).infer_objects()
    return timetable, j, k
timedf, j, k = generate_daily_timetable()
# print(timedf.shape[0])
# timedf.head() ##NOTE THE K_END VALUE IS NO LONGER EQUAL TO THE NEXT MIDPOINT BUT OFFSET BY 1


# %%
# df


# %%
from itertools import chain
def get_daily_j(id=-1):
    jtimes = pd.DataFrame(list(chain.from_iterable(pd.date_range(timedf["j_start"],timedf["midpoint"]) for _,timedf in timedf.iterrows())), columns=("date",))
    jtimes['date'] = jtimes['date'].astype('datetime64[ns]')
    jtimes['period'] = (jtimes['date'].isin(timedf['midpoint'])).shift(1).cumsum().fillna(0).astype(int)
    df_ = df.copy()
    if 'date' not in df_.columns: 
        df_['date'] = pd.to_datetime(df_['Date'])
    df_['date'] = pd.to_datetime(df_['date'])
    df_ = df_.merge(jtimes, how='inner', left_on='date', right_on='date')
    if id==-1: 
        return df_ # this filters out periods that dont end prettily
    else:
        return df_[df_['period']==id].drop(columns='period')
# get_j_df = get_daily_j
def get_daily_k(id=-1):
    ktimes = pd.DataFrame(list(chain.from_iterable(pd.date_range(timedf["midpoint"],timedf["k_end"]) for _,timedf in timedf.iterrows())), columns=("date",))
    ktimes['date'] = ktimes['date'].astype('datetime64[ns]')
    ktimes['period'] = (ktimes['date'].isin(timedf['k_end'])).shift(1).cumsum().fillna(0).astype(int)
    df_ = df.copy()
    if 'date' not in df_.columns: 
        df_['date'] = pd.to_datetime(df_['Date'])
    df_['date'] = pd.to_datetime(df_['date'])
    df_ = df_.merge(ktimes, how='inner', left_on='date', right_on='date')
    if id==-1: 
        return df_ # this filters out periods that dont end prettily
    else:
        return df_[df_['period']==id].drop(columns='period')
# get_daily_j()#.period.plot()


# %%
# get_daily_j(0)


# %%
# get_daily_j()#.period.plot()


# %%
# def generate_timetable(j=1, k=2):
#     '''aims to generate the MONTHLY table of times for each observe/hold period. The midpoint in this context is where we switch from the observation period len=j to the holding period len=k'''
    
#     j_end = pd.date_range(st+pd.DateOffset(months=j*3), et, freq=str(k*3)+"MS") # the 'midpoints' for each observe and hold period. 
#     j_start = j_end + pd.DateOffset(months=-j*3) # based on the midpoint, get the start point for each observe and hold period
#     k_end = j_end + pd.DateOffset(months=k*3) # based on the midpoint, get the end point for each observe and hold period
#     timetable = pd.DataFrame({'j_start': j_start, 'midpoint': j_end, 'k_end': k_end}).infer_objects()
#     return timetable, j, k
# timedf, j, k = generate_timetable(3, 2)
# timedf.head()


# %%
# df['Date']


# %%
# midpoint = timedf.iloc[0].midpoint


# %%
# def get_j_df(id=-1, j=j):
#     '''this should spit out a dataframe of data in the observational period j given specific id or midpoint value. 
#     if unspecified it throws all of them at you with an extra identifying column "periods"'''
#     if (id==-1): # not really sure why we need this but ill include it. this adds a sector column to the data for future filtering purposes if needed
#         df_i = pd.DataFrame(columns=df.columns.to_list()+['period']) #dummy empty df
#         for i in timedf.index: 
#             # print(timedf.iloc[i].j_start,timedf.iloc[i].midpoint)
#             df_ = df[(timedf.iloc[i].j_start <= pd.to_datetime(df['Date']))&(pd.to_datetime(df['Date']) <= timedf.iloc[i].midpoint)] # gets dates btw start and midpt
#             df_['period'] = i
#             df_i = pd.concat([df_i, df_])

#         return df_i
#     return df[(timedf.iloc[id].j_start <= pd.to_datetime(df['Date']))&(pd.to_datetime(df['Date']) <= timedf.iloc[id].midpoint)] # gets dates btw start and midpt from table
# get_j_df(0)


# %%
# def get_k_df(id=-1, k=k):
#     '''this should spit out a dataframe of data in the holding period k given specific id or midpoint value. 
#     if unspecified it throws all of them at you with an extra identifying column "periods"'''
#     if (id==-1): # we can decide if this is useful later this adds a sector column to the data for future filtering purposes if needed
#         df_i = pd.DataFrame(columns=df.columns.to_list()+['period']) #dummy empty df
#         for i in timedf.index: 
#             # print(timedf.iloc[i].j_start,timedf.iloc[i].midpoint)
#             df_ = df[(timedf.iloc[i].midpoint <= pd.to_datetime(df['Date']))&(pd.to_datetime(df['Date']) <= timedf.iloc[i].k_end)] # gets dates btw midpt and end
#             df_['period'] = i
#             df_i = pd.concat([df_i, df_])

#         return df_i
#     return df[(timedf.iloc[id].midpoint <= pd.to_datetime(df['Date']))&(pd.to_datetime(df['Date']) <= timedf.iloc[id].k_end)] # gets dates btw midpt and end
# get_k_df()

# %% [markdown]
# # Data Analysis: using the above functions for building portfolios

# %%
timedf,j,k = generate_daily_timetable(j=20, k=20) # with j as 3 and k as 2
timedf


# %%
df0 = get_daily_j(0).set_index('Name') # i dont know why, but you have to set index to name for the groupby's to work
df0['pct_change'] = df0['Adj Close'].groupby('Name').pct_change()
df0['cum_return'] = (df0['pct_change']+1).groupby('Name').cumprod().fillna(1)-1
df0['adj_close_shifted'] = df0['Adj Close'].groupby('Name').shift(3)#.bfill(0) # filing the value with the backfill TODO check this later
df0['adj_change'] = df0['Adj Close']/df0['adj_close_shifted'] #.bfill(0) 

features = df0.groupby('Name').tail(1).sort_values(['cum_return'], ascending=False)
features


# %%
# monthly_sp500['PYPL'].head() # TODO we should write about some of the issues of yahoofinance as a datasource, that should be sufficient


# %%
# this calculates the winner and loser tickers given n
n = 50
get_percents = lambda n: features.shape[0]//n+1 # this function gets us n percent number of tickers
winner_tickers = features[:get_percents(n)].index
winner_tickers


# %%
loser_tickers = features[-get_percents(n):].index
loser_tickers


# %%
# equally weight
weights = np.ones(get_percents(n))/(get_percents(n))


# %%
winner_df0 = df0.loc[winner_tickers.to_list()]
winner_df0 = winner_df0.pivot_table(index='Date',columns='Name')['pct_change'].fillna(0) # 
winner_df0
# winner_df0.dot(weights) # gives us the percent change of the portfolios
winner_performance = (winner_df0.dot(weights)+1).cumprod() - 1 # gets cumulative return for a period
winner_performance


# %%
def get_cum_return(data):
    '''gets cumulative return based on adjusted closing price of all tickers in input dataframe'''
    df_ = data.set_index('Name')

    # df_['pct_change'] = df_['Adj Close'].groupby('Name').pct_change()
    df_['pct_change'] = df_['Adj Close'].groupby('Name').pct_change()
    df_['cum_return'] = (df_['pct_change']+1).groupby('Name').cumprod().fillna(1)-1
    df_['adj_close_shifted'] = df_['Adj Close'].groupby('Name').shift(3)#.bfill(0) # filing the value with the backfill TODO check this later
    df_['adj_change'] = df_['Adj Close']/df_['adj_close_shifted'] #.bfill(0) 
    df_['adj_close_shifted'] = df_['Adj Close'].groupby('Name').shift(3)#.bfill(0) # filing the value with the backfill TODO check this later
    df_['adj_change'] = df_['Adj Close']/df_['adj_close_shifted'] #.bfill(0) 
    
    return df_

def get_percents(n): 
    '''this function gets us n percent number of tickers'''
    return features.shape[0]//n+1

def get_portfolios(period=0, n=50):   
    '''returns the portfolios we want from a particular period'''
    ## Gets the portfolio(s) in question we want to look at
    data = get_cum_return(get_daily_j(period))
    features = data.groupby('Name').tail(1).sort_values(['cum_return'], ascending=False) # sorts tickers by cumulative return

    winner_tickers = features[:get_percents(n)].index # we get the top/bottom n percent tickers
    loser_tickers = features[-get_percents(n):].index # only winner and loser portfolios for now, we could expand later
    # TODO add more portfolios if we have time
    # print(len(winner_tickers), len(loser_tickers))
    print('.', end='')
    return (winner_tickers.to_list(), loser_tickers.to_list())

def get_portfolio_performance(period=0, n=2, weights=None, hold=False):
    '''Gets all portfolios' cumulative return performance based on n period 
    keyword args:
    period      -- the nth period of data we are looking at, default 0
    portfolio   -- the particular type of portfolio we want to be looking at (winner or loser, etc) default winner/momentum
    n           -- the percentage of tickers we want to be looking at
    weights     -- how to weight the portfolio values. if unspecified (None) we assume equal weighting in the portfolio
    hold        -- if true returns the performance evaluation of the holding period k, else returns the performance of the observation period j
    '''
    winner_tickers, loser_tickers = get_portfolios(period, n)

    ## evaluates the performance of portfolios on either hold or observational data
    eval_df = get_cum_return(get_k_df(period)) if hold else get_cum_return(get_daily_j(period))# we get the return from the hold period
    
    weights = np.ones(get_percents(n))/(get_percents(n)) if not weights else weights # set weights
    print('weights', len(weights))
    winner_eval = eval_df.loc[winner_tickers] # first the winners
    winner_eval = winner_eval.pivot_table(index='Date',columns='Name')['pct_change'].fillna(0)
    winner_performance = (winner_eval.dot(weights)+1).cumprod() - 1 # cumulative return

    loser_eval = eval_df.loc[loser_tickers] # then the losers
    loser_eval = loser_eval.pivot_table(index='Date',columns='Name')['pct_change'].fillna(0)
    loser_performance = (loser_eval.dot(weights)+1).cumprod() - 1 # cumulative return

    output = pd.DataFrame({'winners': winner_performance, 'losers': loser_performance})
    return output
get_portfolio_performance()


# %%
get_portfolios(1)


# %%
# 10/500
# pf = pd.DataFrame({'idx':timedf.index})
# pf['mom'], pf['rev'] = zip(*pf['idx'].map(get_portfolios))
# pf


# %%
portfolios = {'mom':{}, 'rev':{}}
for i in timedf.index:
    # print(i)
    mom, rev = get_portfolios(i)
    portfolios['mom'][i] = mom
    portfolios['rev'][i] = rev
# portfolios['mom'][0]
pf = timedf.merge(pd.DataFrame(portfolios), left_index=True, right_index=True)
pf.head()


# %%
def get_portfolio_pct(lst, date, weights=None):
    '''given list of tickers, generates a EQUAL WEIGHTED INDEX portfolio and aggregates their performance'''
    portfolio_df = df[df['Name'].isin(lst)] #filters df by lst
    portfolio_df = get_cum_return(portfolio_df)[['Date','pct_change']] # we get cum return
    portfolio_df = portfolio_df.groupby('Date').mean()
    value = portfolio_df[pd.to_datetime(portfolio_df.index)==pd.to_datetime(date)].values
    # print('.', end='')
    return 0 if (value.shape[0]==0 or np.isnan(value[0,0])) else value[0,0] # will need to spend time staring at the output to see if most columns are good
    # im thinking the only good value we can get out of this is percent change and maybe cum_return, ohlc columns are garbage
# temp = get_portfolio_pct(['NFLX', 'LVS', 'URI', 'CMG', 'FFIV'], dpf.Date[0])
# temp


# %%
dpf = pd.DataFrame({'Date':pd.date_range(df.Date.min(), df.Date.max(), freq="D")}).merge(pf[['k_end', 'mom', 'rev']].reset_index(), how='left', left_on='Date', right_on='k_end').bfill().ffill()
dpf['mom_pct_change'] = dpf.apply(lambda x: get_portfolio_pct(x.mom, x.Date), axis=1)
dpf['rev_pct_change'] = dpf.apply(lambda x: get_portfolio_pct(x.rev, x.Date), axis=1)
dpf['index'] = dpf['index'].astype(int)
dpf = dpf.drop(['mom', 'rev', 'k_end'], axis=1)
dpf['mom_return'] = ((dpf['mom_pct_change']+1).cumprod()-1) * 100
dpf['rev_return'] = ((dpf['rev_pct_change']+1).cumprod()-1) * 100
dpf[['mom_return', 'rev_return', 'Date']].set_index('Date').plot(figsize=(12,8))
dpf


# %%
spc = pdr.get_data_yahoo("^GSPC", start="2009-10-01")
spc


# %%
import matplotlib.ticker as mtick

# fig = plt.figure(1, (12,8))
# ax = fig.add_subplot(1,1,1)
# fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
# xticks = mtick.FormatStrFormatter(fmt)
# ax.yaxis.set_major_formatter(xticks)
spc['pctc'] = spc['Adj Close'].pct_change()
spc['cumprod'] =((spc['pctc']+1).cumprod()-1)*100
sp500 = spc.merge(dpf, how='left', left_index=True, right_on='Date')
# sp500[['Date', 'mom_return', 'rev_return','cumprod']].set_index('Date').plot.area(ax=ax,stacked=False,figsize=(12,8)) # plotted
# ax.legend(["Momentum", "Reversal", "S&P500"])
# fig.savefig('performance.png')


# %%
sp500 = sp500.drop(['mom_return', 'rev_return'], axis=1) # mom_retunr and rev_return, beware of use
sp500 = sp500.reset_index(drop=True).set_index('Date')

sp500.ta.cci(length=4,append=True)
sp500.ta.kdj(length=4,append=True)
sp500.ta.rsi(length=4,append=True)
sp500.ta.bop(length=4,append=True)
sp500.ta.willr(length=4,append=True)
sp500.ta.pdist(length=4,append=True)
sp500.ta.kc(length=4,append=True)
sp500.ta.adx(length=4,append=True)
sp500.ta.qstick(length=4,append=True)
sp500.ta.roc(length=4,append=True)
# sp500.ta.ao(length=4,append=True)
# sp500.ta.macd(fast=4,slow=8,append=True)
sp500.ta.stdev(length=4,append=True)
sp500.ta.pvol(append=True)
sp500.ta.efi(length=4, append=True)
sp500 = sp500[sp500.index.isin(pd.date_range(timedf.j_start.min(), timedf.k_end.max()))]
sp500


# %%
get_j_df = get_daily_j
get_k_df = get_daily_k
# get_j_df().groupby('period').count()


# %%
sp500 = sp500.dropna().reset_index()
# sp500

# %% [markdown]
# # START HERE! avoid most everything above

# %%
# df = pd.read_csv('SP500_daily.csv').infer_objects().dropna()
# def pull_data(j=10,k=20):
#     # this runs on bad code. Please restart the kernel and run all if you run into issues.
#     tdf, j, k = generate_daily_timetable(j=j,k=k)
#     df_ = pd.read_csv('daily_j'+str(j)+'_k'+str(k)+'.csv')
#     return tdf, df_, j, k
# timedf, df, j, k= pull_data(30,60)
df=sp500
timedf


# %%
df


# %%
df['index'] = df['index'].astype(int)
df # all the features you neeeed


# %%
timedf

# %% [markdown]
# # Charting

# %%
# from matplotlib.dates import date2num
# plotdf = df[df['index'] < 5][['index', 'date']]
# plotdf['date'] = pd.to_datetime(plotdf['date'])

# ax = plotdf.set_index('date').plot(figsize=(8, 5), title='Observe and Hold Periods (j=10, k=20 days)', yticks=[0,1,2,3,4], xlabel='Date', ylabel='Period')
# ax.lines.pop(0)

# for i in range(5):
#     td = timedf.iloc[i]
    
#     # ax.axvspan(td.j_start.strftime("%Y-%m-%d"), td.midpoint.strftime("%Y-%m-%d"), color='green', alpha=0.3)
#     # ax.axvspan(td.midpoint.strftime("%Y-%m-%d"), td.k_end.strftime("%Y-%m-%d"), color='grey', alpha=0.3)
#     ax.hlines(y=i, xmin=td.j_start.strftime("%Y-%m-%d"), xmax=td.midpoint.strftime("%Y-%m-%d"), linewidth=15, color='#d3d3d3', label='Observation period' if i==0 else '_nolegend_')
#     ax.hlines(y=i,xmin=td.midpoint.strftime("%Y-%m-%d"), xmax=td.k_end.strftime("%Y-%m-%d"), linewidth=15, color='grey', label='Hold period' if i==0 else '_nolegend_')

# ax.legend(loc='upper left')#.remove()
# plt.show();


# %%
timedf.iloc[0].j_start.strftime("%Y-%m-%d")
timedf.iloc[0].midpoint.strftime("%Y-%m-%d")


# %%
j0 = get_daily_j()
j0

print('done with features!')
# %%
def featurize(inputdf):
    df_ = inputdf.copy()
    df_['log_ret'] = np.log(df_['adj_close']) - np.log(df_['adj_close'].shift(1))
    df_['log_ret'] = df_['log_ret'].fillna(0)
    df_['adj_close_diff'] = df_['adj_close'].diff()
    df_['pctc'] = df_['adj_close'].pct_change().fillna(0)

    # df_['shifted_mom_pct_change'] = df_.mom_pct_change.shift()
    # df_['shifted_rev_pct_change'] = df_.rev_pct_change.shift()
    # df_['shifted_sp_pct_change'] = df_.pctc.shift()

    df_['cumprod'] = (df_['pctc']+1).cumprod()-1
    df_['mom_cumprod'] = (df_['mom_pct_change']+1).cumprod()-1
    df_['rev_cumprod'] = (df_['rev_pct_change']+1).cumprod()-1
    df_['SMA_5'] = df_['adj_close'].rolling(window=5).mean().fillna(method='bfill').fillna(0) #NOTE: changed from 10 to 5
    def label(mom_pct_ret,rev_pct_ret):
        if(mom_pct_ret > 0 and mom_pct_ret >= rev_pct_ret):
            return 1
        elif(mom_pct_ret < rev_pct_ret and rev_pct_ret > 0):
            return -1
        else:
            return 0

    def label_condition(rsi):
        if(rsi > 70.0):
            return 'OVERBOUGHT'
        elif(rsi < 30.0):
            return 'OVERSOLD'
        else:
            return 'NEUTRAL'
    # df_['movement'] = df_.apply(lambda x: label(x.mom_pct_change, x.rev_pct_change), axis=1).astype(int) # @amir i renamed your labels since i wasnt sure what this was
    df_['condition_OVERBOUGHT'] = (df['RSI_4'] > 70).astype(int)
    df_['condition_OVERSOLD'] = (df['RSI_4'] < 30).astype(int)
    df_['condition_NEUTRAL'] = ((df['RSI_4'] >= 30) & (df['RSI_4'] <= 70)).astype(int)
    label_value = 1 if df_.mom_cumprod.iloc[-1] > 0 and df_.mom_cumprod.iloc[-1] >= df_.rev_cumprod.iloc[-1] else -1 if df_.mom_cumprod.iloc[-1] < 0 and df_.rev_cumprod.iloc[-1] > 0 else 0 # returns the label value

    feat_vector = df_.agg( # here's the feature vector, a real chonker
        #encoded distribution information
        adj_close_mean=('adj_close',np.mean),
        adj_close_min =('adj_close',min),
        adj_close_max =('adj_close',max),
        adj_close_std =('adj_close',np.std),
        adj_close_diff_mean=('adj_close_diff',np.mean),
        adj_close_diff_min =('adj_close_diff',min),
        adj_close_diff_max =('adj_close_diff',max),
        adj_close_diff_std =('adj_close_diff',np.std),
        pct_chng_mean=('pctc',np.mean),
        pct_chng_min =('pctc',min),
        pct_chng_max =('pctc',max),
        pct_chng_std =('pctc',np.std),
        volume_mean=('volume',np.mean),
        volume_min =('volume',min),
        volume_max =('volume',max),
        volume_std =('volume',np.std),
        KCBe_4_2_mean=('KCBe_4_2',np.mean),
        KCBe_4_2_min =('KCBe_4_2',min),
        KCBe_4_2_max =('KCBe_4_2',max),
        KCBe_4_2_std =('KCBe_4_2',np.std),
        CCI_mean=('CCI_4_0.015',np.mean),
        CCI_min =('CCI_4_0.015',min),
        CCI_max =('CCI_4_0.015',max),
        CCI_std =('CCI_4_0.015',np.std),
        K_4_3_mean=('K_4_3',np.mean),
        K_4_3_min =('K_4_3',min),
        K_4_3_max =('K_4_3',max),
        K_4_3_std =('K_4_3',np.std),
        D_4_3_mean=('D_4_3',np.mean),
        D_4_3_min =('D_4_3',min),
        D_4_3_max =('D_4_3',max),
        D_4_3_std =('D_4_3',np.std),
        J_4_3_mean=('J_4_3',np.mean),
        J_4_3_min =('J_4_3',min),
        J_4_3_max =('J_4_3',max),
        J_4_3_std =('J_4_3',np.std),
        RSI_4_mean=('RSI_4',np.mean),
        RSI_4_min =('RSI_4',min),
        RSI_4_max =('RSI_4',max),
        RSI_4_std =('RSI_4',np.std),
        BOP_mean=('BOP',np.mean),
        BOP_min =('BOP',min),
        BOP_max =('BOP',max),
        BOP_std =('BOP',np.std),
        PDIST_mean=('PDIST',np.mean),
        PDIST_min =('PDIST',min),
        PDIST_max =('PDIST',max),
        PDIST_std =('PDIST',np.std),
        KCLe_4_2_mean=('KCLe_4_2',np.mean),
        KCLe_4_2_min =('KCLe_4_2',min),
        KCLe_4_2_max =('KCLe_4_2',max),
        KCLe_4_2_std =('KCLe_4_2',np.std),
        ADX_4_mean=('ADX_4',np.mean),
        ADX_4_min =('ADX_4',min),
        ADX_4_max =('ADX_4',max),
        ADX_4_std =('ADX_4',np.std),
        DMP_4_mean=('DMP_4',np.mean),
        DMP_4_min =('DMP_4',min),
        DMP_4_max =('DMP_4',max),
        DMP_4_std =('DMP_4',np.std),
        DMN_4_mean=('DMN_4',np.mean),
        DMN_4_min =('DMN_4',min),
        DMN_4_max =('DMN_4',max),
        DMN_4_std =('DMN_4',np.std),
        QS_4_mean=('QS_4',np.mean),
        QS_4_min =('QS_4',min),
        QS_4_max =('QS_4',max),
        QS_4_std =('QS_4',np.std),
        ROC_4_mean=('ROC_4',np.mean),
        ROC_4_min =('ROC_4',min),
        ROC_4_max =('ROC_4',max),
        ROC_4_std =('ROC_4',np.std),
        STDEV_4_mean=('STDEV_4',np.mean),
        STDEV_4_min =('STDEV_4',min),
        STDEV_4_max =('STDEV_4',max),
        STDEV_4_std =('STDEV_4',np.std),
        PVOL_mean=('PVOL',np.mean),
        PVOL_min =('PVOL',min),
        PVOL_max =('PVOL',max),
        PVOL_std =('PVOL',np.std),
        EFI_4_mean=('EFI_4',np.mean),
        EFI_4_min =('EFI_4',min),
        EFI_4_max =('EFI_4',max),
        EFI_4_std =('EFI_4',np.std),
        SMA_5_mean=('SMA_5',np.mean),
        SMA_5_min =('SMA_5',min),
        SMA_5_max =('SMA_5',max),
        SMA_5_std =('SMA_5',np.std),
        log_ret_mean=('log_ret',np.mean),
        log_ret_min =('log_ret',min),
        log_ret_max =('log_ret',max),
        log_ret_std =('log_ret',np.std),
        # getting last values
        cumprod_last=('cumprod', lambda x: x.iloc[-1]),
        mom_cumprod_last=('mom_cumprod', lambda x: x.iloc[-1]),
        rev_cumprod_last=('rev_cumprod', lambda x: x.iloc[-1]),
        log_ret_last=('log_ret', lambda x: x.iloc[-1]),
        #getting ratios. Watch for zero division errors.
        # movement_sell_mom_ratio=('movement', lambda x: (x==0).sum()/(x==1).sum()),
        # movement_rev_mom_ratio=('movement', lambda x: (x==-1).sum()/(x==1).sum()), # this breaks the function for some reason lol
        ).stack()
    feat_vector.index = feat_vector.index.droplevel(1) # dealing with multiindex shenanigans
    feat_vector = feat_vector.append(pd.Series({'label':label_value}))
    return feat_vector#, df_
# featvect0, outputdf = featurize(j0)
# featvect0
features = j0.groupby('index').apply(featurize)
features

# %% [markdown]
# # train your models on this data ^^

# %%
print('j: ',j,' k: ',k)


# %%
# features.to_csv('daily_j10_k20_features.csv')
# features.to_csv('daily_j20_k40_features.csv')
# features.to_csv('daily_j10_k10_features.csv')
features.to_csv('daily_j'+str(j)+'_k'+str(k)+'_features.csv')


# %%
# timedf


# %%