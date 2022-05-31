# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 11:23:25 2022

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 00:41:50 2022

@author: HP
"""

# =============================================================================
#Introduction: - 
#This trading strategy is to find the Supply and Demand Zond on based of the 
# Boring Candle and Exciting Candle
# 
# Boring Candle :- have less than 50% of the body. (Diff between opening and 
# Closing price is less than 50% of Diff betwee high and low of the Candle)
# 
# Exciting Candle (Leg Candle):- have more than 50% of the body. (Diff between
# opening and Closing price is more than 50% of Diff between high 
# and low of the Candle)
# 
# Exiting Candles are of two Type: 1- Legin Candle & 2. Legout Candle.
#
# in this strategy we look for two to three boring candle at a place, which
# is called base.
# After base exciting candle is called Legout Candle
# Before base exciting candle is called Legin Candle
# 
# We use all this information at Lower TimeFrame for Entry and Exit price.
# I use MACD for Trend Analysis at Higher Time Frame
#
# =============================================================================
# Challenges: - Trader has to put so much effors to find the zone in this strategy
# he has to open every stock and look for the combination of Base and Leg Candles
# and moreover Trader get very less trade in this strategy, but 80-90% of the trade
# gives profit.
# by automation/using this tool trader can look into 100's of stock in 2 minutes 
# and can plan their Trade.
# =============================================================================


# As Differnt Time frames have differnt combination of Lower Time Frame and 
# Higher Time Frame, must be update here for any other time frame

ltf = '15m'
htf='1d'

import numpy as np
import pandas as pd
import yfinance as yf  # to access the financial data available on Yahoo Finance
import datetime as dt
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm  # Help in to get the slope

# Change number in timedelta following as how many Days of data required,

today = dt.datetime.today()
end = dt.datetime.today()
start = end-dt.timedelta(7)

datestamp=pd.to_datetime(today,format='%Y/%m/%d')

# create function to get the Average True Range, Average True range
# max (Difference between current High & Low, current High and Previous Close, 
# current low and preivous close).

def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Adj Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Adj Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)/df['Adj Close']
    df['ATR'] = df['TR'].rolling(n).mean()
#    df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()/df['Adj Close']
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

# strock Tickers :- in these stocks i am interested to trade

tickers=['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
        'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
        'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS',
        'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS',
        'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
        'HINDUNILVR.NS', 'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'IOC.NS',
        'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
        'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS',
        'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS',
        'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATACONSUM.NS',
        'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 'TITAN.NS', 'UPL.NS',
        'ULTRACEMCO.NS', 'WIPRO.NS', 'ACC.NS','ADANIENT.NS','ADANIGREEN.NS',
        'ADANITRANS.NS','AMBUJACEM.NS','APOLLOHOSP.NS','AUROPHARMA.NS',
        'DMART.NS','BAJAJHLDNG.NS','BANDHANBNK.NS','BANKBARODA.NS',
        'BERGEPAINT.NS','BIOCON.NS','BOSCHLTD.NS', 'CHOLAFIN.NS','COLPAL.NS',
        'DLF.NS','DABUR.NS','GAIL.NS','GLAND.NS', 'GODREJCP.NS',
        'HDFCAMC.NS', 'HAVELLS.NS','HINDPETRO.NS','ICICIGI.NS',
        'ICICIPRULI.NS','IGL.NS','INDUSTOWER.NS','NAUKRI.NS','INDIGO.NS',
        'JINDALSTEL.NS','JUBLFOOD.NS','LTI.NS','LUPIN.NS','MARICO.NS',
        'MUTHOOTFIN.NS','NMDC.NS','PIIND.NS','PIDILITIND.NS','PEL.NS',
        'PGHH.NS','PNB.NS','SBICARD.NS','SIEMENS.NS','SAIL.NS',
        'TORNTPHARM.NS','MCDOWELL-N.NS','VEDL.NS']

# Access Data at Lower Time Frame

ohlc_ltf={}
for ticker in tickers:
    temp=yf.download(ticker,start,end,interval=ltf)
    temp.dropna(how='any',inplace=True)
    ohlc_ltf[ticker]=temp

# Make the Candle column for category of candles

for ticker in tickers:
    ohlc_ltf[ticker]['Candle']=''
    for i in range(len(ohlc_ltf[ticker])):
        if (abs(ohlc_ltf[ticker]['Open'][i]-ohlc_ltf[ticker]['Adj Close'][i]))/(ohlc_ltf[ticker]['High'][i]-ohlc_ltf[ticker]['Low'][i])>.51:
            ohlc_ltf[ticker]['Candle'][i]='Leg Candle'
        else:
            ohlc_ltf[ticker]['Candle'][i]='Boring Candle'

# Create a column to show differnce betwee closing price of previous candle & current candle in percentage

for ticker in tickers:        
    for i in range(1,len(ohlc_ltf[ticker]["Adj Close"])):
        ohlc_ltf[ticker]['diff_price']=ohlc_ltf[ticker]['Adj Close'].pct_change()

# Using the Diff price column change the column with Green Candle or Red candle follwoing Price increase of decrease respectively.

for ticker in tickers:        
    for i in range(len(ohlc_ltf[ticker])):
        if ohlc_ltf[ticker]['Candle'][i]=='Leg Candle' and ohlc_ltf[ticker]['diff_price'][i]>0:
            ohlc_ltf[ticker]['Candle'][i]='Green Candle'
        elif ohlc_ltf[ticker]['Candle'][i]=='Leg Candle' and ohlc_ltf[ticker]['diff_price'][i]<0:
            ohlc_ltf[ticker]['Candle'][i]='Red Candle'

# Using ATR Function create two new columns for Average True and True Range.

for ticker in tickers:  
    ohlc_ltf[ticker]['ATR']=ATR(ohlc_ltf[ticker],20)['ATR']
    ohlc_ltf[ticker]['TR']=ATR(ohlc_ltf[ticker],20)['TR']

# Drop the Nan values

for ticker in tickers:
    ohlc_ltf[ticker].dropna(inplace=True)

# make a new column min range using ATR and TR value, take min value from TR and ATR.
for ticker in tickers:
    ohlc_ltf[ticker]['min_range']=''
    ohlc_ltf[ticker].reset_index(inplace=True)
    for i in range(len(ohlc_ltf[ticker])):
        ohlc_ltf[ticker]['min_range'][i]=min( ohlc_ltf[ticker]['ATR'][i], ohlc_ltf[ticker]['TR'][i])

# Further divide Candles in Green Legin, Green Legout, Red Legin & Red Legout and uses some calculation of practical bases/hyper parameter tuning

for ticker in tickers:         
    ohlc_ltf[ticker]['Candle_cat']=''          
    for i in range(len(ohlc_ltf[ticker]['Candle'])):
        if ohlc_ltf[ticker]['Candle'][i]=='Boring Candle':
            ohlc_ltf[ticker]['Candle_cat'][i]='Boring Candle'
        elif ohlc_ltf[ticker]['Candle'][i]=='Green Candle' and abs(ohlc_ltf[ticker]['diff_price'][i])>=1.4*ohlc_ltf[ticker]['min_range'][i]:
            ohlc_ltf[ticker]['Candle_cat'][i]='Green Legout'
        elif ohlc_ltf[ticker]['Candle'][i]=='Green Candle' and abs(ohlc_ltf[ticker]['diff_price'][i])<1.4*ohlc_ltf[ticker]['min_range'][i] and abs(ohlc_ltf[ticker]['diff_price'][i])>.33*ohlc_ltf[ticker]['min_range'][i]:
            ohlc_ltf[ticker]['Candle_cat'][i]='Green Legin'
        elif ohlc_ltf[ticker]['Candle'][i]=='Red Candle' and abs(ohlc_ltf[ticker]['diff_price'][i])>=1.4*ohlc_ltf[ticker]['min_range'][i]:
            ohlc_ltf[ticker]['Candle_cat'][i]='Red Legout'
        elif ohlc_ltf[ticker]['Candle'][i]=='Red Candle' and abs(ohlc_ltf[ticker]['diff_price'][i])<1.4*ohlc_ltf[ticker]['min_range'][i] and abs(ohlc_ltf[ticker]['diff_price'][i])>.33*ohlc_ltf[ticker]['min_range'][i]:
            ohlc_ltf[ticker]['Candle_cat'][i]='Red Legin'

# create a column which gives the increase the number continously for same kind of candles becuase more than 4 boring candle
# is of no use, or dilute idea of base

for ticker in tickers:
    ohlc_ltf[ticker]["bar_num"] = np.where(ohlc_ltf[ticker]["Candle"]=='Boring Candle',1,0)
  
# White Area - Price does not reverse to zone yet, here white are part implmented.
for ticker in tickers:
    for i in range(1,len(ohlc_ltf[ticker]["bar_num"])):
        if ohlc_ltf[ticker]["bar_num"][i]>0 and ohlc_ltf[ticker]["bar_num"][i-1]>0:
            ohlc_ltf[ticker]["bar_num"][i]+=ohlc_ltf[ticker]["bar_num"][i-1]
    ohlc_ltf[ticker]['white area']=''
    for i in range(1,len(ohlc_ltf[ticker])):
        if ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['Adj Close'][i-1]*.9985<ohlc_ltf[ticker]['Low'][i]:
            ohlc_ltf[ticker]['white area'][i]='yes'
        elif ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['High'][i]*.9985<ohlc_ltf[ticker]['Low'][i-1]:
            ohlc_ltf[ticker]['white area'][i]='yes'
        
# create the Date column

for ticker in tickers:
    ohlc_ltf[ticker].reset_index(drop=True,inplace=True)
    ohlc_ltf[ticker]['date']=''
    for i in range(len(ohlc_ltf[ticker]['Datetime'])):
        ohlc_ltf[ticker]['date'][i]=ohlc_ltf[ticker]['Datetime'][i].date()

# Finally Here using all information Buy and Sell signal coded using LTF only        
for ticker in tickers:
    ohlc_ltf[ticker]['signal']=''
    for i in range(6,len(ohlc_ltf[ticker]["Candle_cat"])):
        if (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==1 and ohlc_ltf[ticker]['Candle_cat'][i-2]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-2]):
            ohlc_ltf[ticker]['signal'][i]='buy'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==2 and ohlc_ltf[ticker]['Candle_cat'][i-3]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-3]):
            ohlc_ltf[ticker]['signal'][i]='buy'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==3 and ohlc_ltf[ticker]['Candle_cat'][i-4]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-4]):
            ohlc_ltf[ticker]['signal'][i]='buy' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==4 and ohlc_ltf[ticker]['Candle_cat'][i-5]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-5]):
            ohlc_ltf[ticker]['signal'][i]='buy' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==1 and ohlc_ltf[ticker]['Candle_cat'][i-2]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-2]):
            ohlc_ltf[ticker]['signal'][i]='buy'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==2 and ohlc_ltf[ticker]['Candle_cat'][i-3]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-3]):
            ohlc_ltf[ticker]['signal'][i]='buy'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==3 and ohlc_ltf[ticker]['Candle_cat'][i-4]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-4]):
            ohlc_ltf[ticker]['signal'][i]='buy' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Green Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==4 and ohlc_ltf[ticker]['Candle_cat'][i-5]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-5]):
            ohlc_ltf[ticker]['signal'][i]='buy' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==1 and ohlc_ltf[ticker]['Candle_cat'][i-2]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-2]):
            ohlc_ltf[ticker]['signal'][i]='sell'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==2 and ohlc_ltf[ticker]['Candle_cat'][i-3]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-3]):
            ohlc_ltf[ticker]['signal'][i]='sell'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==3 and ohlc_ltf[ticker]['Candle_cat'][i-4]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-4]):
            ohlc_ltf[ticker]['signal'][i]='sell' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==4 and ohlc_ltf[ticker]['Candle_cat'][i-5]=='Red Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-5]):
            ohlc_ltf[ticker]['signal'][i]='sell' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==1 and ohlc_ltf[ticker]['Candle_cat'][i-2]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-2]):
            ohlc_ltf[ticker]['signal'][i]='sell'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==2 and ohlc_ltf[ticker]['Candle_cat'][i-3]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-3]):
            ohlc_ltf[ticker]['signal'][i]='sell'
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==3 and ohlc_ltf[ticker]['Candle_cat'][i-4]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-4]):
            ohlc_ltf[ticker]['signal'][i]='sell' 
        elif (ohlc_ltf[ticker]['Candle_cat'][i]=='Red Legout' and ohlc_ltf[ticker]['bar_num'][i-1]==4 and ohlc_ltf[ticker]['Candle_cat'][i-5]=='Green Legin') and (ohlc_ltf[ticker].date[i]==ohlc_ltf[ticker].date[i-1]==ohlc_ltf[ticker].date[i-5]):
            ohlc_ltf[ticker]['signal'][i]='sell' 



# Higher Time Frame Data Access

endhtf = dt.datetime.today()
starthtf = endhtf-dt.timedelta(90)

ohlc_htf_copy={}
for ticker in tickers:
    temp=yf.download(ticker,starthtf,endhtf,interval=htf)
    temp.dropna(how='any',inplace=True)
    ohlc_htf_copy[ticker]=temp

# Create Function for MACD

def MACD(DF,a=12,b=26,c=9):
    df=DF.copy() # to avoid change in original dataframe
    df['ma_fast']=df['Adj Close'].ewm(span=a,min_periods=a).mean()
    df['ma_slow']=df['Adj Close'].ewm(span=b,min_periods=b).mean()
    df['macd']=df['ma_fast']-df['ma_slow']
    df['signal']=df['macd'].ewm(span=c,min_periods=c).mean()
    return (df['macd'],df['signal'])

# Create Function to get slope to compare MACD and Signal Values

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

# Create a deep copy to keep original data safe

ohlc_htf = copy.deepcopy(ohlc_htf_copy)

# Create 4 New Columns as below and get the values using MACD & Slope Function

for ticker in tickers:
    ohlc_htf[ticker]["macd"]= MACD(ohlc_htf[ticker],12,26,9)[0]
    ohlc_htf[ticker]["macd_sig"]= MACD(ohlc_htf[ticker],12,26,9)[1]
    ohlc_htf[ticker]["macd_slope"] = slope(ohlc_htf[ticker]["macd"],5)
    ohlc_htf[ticker]["macd_sig_slope"] = slope(ohlc_htf[ticker]["macd_sig"],5)
    ohlc_htf[ticker].dropna(inplace=True)

# Create Date Column in HTF Data
for ticker in tickers:
    ohlc_htf[ticker].reset_index(inplace=True)
    ohlc_htf[ticker]['date']=''
    for i in range(len(ohlc_htf[ticker]["Date"])):
        ohlc_htf[ticker]['date'][i]=ohlc_htf[ticker].Date[i].date()
        
# Create new empty dictttionary and merge both LTF Data and HTF Data on Date Column   
     
new_dict={}
for ticker in tickers:
    new_dict[ticker]=pd.merge(left=ohlc_ltf[ticker], right = ohlc_htf[ticker], on = 'date', how='inner')

# now create the Real Zone using all info MACD, Signal, White Area columns

for ticker in tickers:
    new_dict[ticker]['Real Zone']=''
    for i in range(len(new_dict[ticker])):
        if new_dict[ticker]['signal'][i]=='buy' and new_dict[ticker]['macd'][i]>new_dict[ticker]['macd_sig'][i] and new_dict[ticker]['macd_slope'][i]>new_dict[ticker]['macd_sig_slope'][i] and new_dict[ticker]['white area'][i]=='yes':
            new_dict[ticker]['Real Zone'][i]='buy'
        elif new_dict[ticker]['signal'][i]=='sell' and new_dict[ticker]['macd'][i]<new_dict[ticker]['macd_sig'][i] and new_dict[ticker]['macd_slope'][i]<new_dict[ticker]['macd_sig_slope'][i] and new_dict[ticker]['white area'][i]=='yes':
            new_dict[ticker]['Real Zone'][i]='sell'

# Delete the row having NaN values
 for ticker in tickers:       
     new_dict[ticker]['Real Zone'].replace({'':np.NaN},inplace=True)
 
# Now create Final Ticker dictionary, which shows the avaiable zone of current date.
  
 Final_ticker={}
 for ticker in tickers:
     Final_ticker[ticker]=new_dict[ticker][new_dict[ticker]['date']==datestamp]
     Final_ticker[ticker]=Final_ticker[ticker].loc[Final_ticker[ticker]['Real Zone'].notnull(),['Datetime','Real Zone']]
 
 
# =============================================================================
# Now Buy Opening the Final Ticker dictionary , if any number at row place size it
# means zone is availabe in that stock open and get the DateTime when the 
# Zone was created and plan to take the trade. This resolve the challenge of the
# trader, save lots of time & increase the accuracy.
# =============================================================================
