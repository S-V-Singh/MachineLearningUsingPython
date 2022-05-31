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
ltf = '15m'
htf='1d'

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from stocktrends import Renko

datestamp=pd.to_datetime('2022/02/23',format='%Y/%m/%d')

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

end = dt.datetime.today()
start = end-dt.timedelta(7)


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
        'BERGEPAINT.NS','BIOCON.NS','BOSCHLTD.NS','CADILAHC.NS',
        'CHOLAFIN.NS','COLPAL.NS','DLF.NS','DABUR.NS','GAIL.NS','GLAND.NS',
        'GODREJCP.NS','HDFCAMC.NS','HAVELLS.NS','HINDPETRO.NS','ICICIGI.NS',
        'ICICIPRULI.NS','IGL.NS','INDUSTOWER.NS','NAUKRI.NS','INDIGO.NS',
        'JINDALSTEL.NS','JUBLFOOD.NS','LTI.NS','LUPIN.NS','MARICO.NS',
        'MUTHOOTFIN.NS','NMDC.NS','PIIND.NS','PIDILITIND.NS','PEL.NS',
        'PGHH.NS','PNB.NS','SBICARD.NS','SIEMENS.NS','SAIL.NS',
        'TORNTPHARM.NS','MCDOWELL-N.NS','VEDL.NS']

# =============================================================================
# tickers=['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS',
#          'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS']
# =============================================================================

ohlc_ltf={}
for ticker in tickers:
    temp=yf.download(ticker,start,end,interval=ltf)
    temp.dropna(how='any',inplace=True)
    ohlc_ltf[ticker]=temp

for ticker in tickers:
    ohlc_ltf[ticker]['Candle']=''
    for i in range(len(ohlc_ltf[ticker])):
        if (abs(ohlc_ltf[ticker]['Open'][i]-ohlc_ltf[ticker]['Adj Close'][i]))/(ohlc_ltf[ticker]['High'][i]-ohlc_ltf[ticker]['Low'][i])>.51:
            ohlc_ltf[ticker]['Candle'][i]='Leg Candle'
        else:
            ohlc_ltf[ticker]['Candle'][i]='Boring Candle'
for ticker in tickers:        
    for i in range(1,len(ohlc_ltf[ticker]["Adj Close"])):
        ohlc_ltf[ticker]['diff_price']=ohlc_ltf[ticker]['Adj Close'].pct_change()

for ticker in tickers:        
    for i in range(len(ohlc_ltf[ticker])):
        if ohlc_ltf[ticker]['Candle'][i]=='Leg Candle' and ohlc_ltf[ticker]['diff_price'][i]>0:
            ohlc_ltf[ticker]['Candle'][i]='Green Candle'
        elif ohlc_ltf[ticker]['Candle'][i]=='Leg Candle' and ohlc_ltf[ticker]['diff_price'][i]<0:
            ohlc_ltf[ticker]['Candle'][i]='Red Candle'

for ticker in tickers:  
    ohlc_ltf[ticker]['ATR']=ATR(ohlc_ltf[ticker],20)['ATR']
    ohlc_ltf[ticker]['TR']=ATR(ohlc_ltf[ticker],20)['TR']

for ticker in tickers:
    ohlc_ltf[ticker].dropna(inplace=True)

for ticker in tickers:
    ohlc_ltf[ticker]['min_range']=''
    ohlc_ltf[ticker].reset_index(inplace=True)
    for i in range(len(ohlc_ltf[ticker])):
        ohlc_ltf[ticker]['min_range'][i]=min( ohlc_ltf[ticker]['ATR'][i], ohlc_ltf[ticker]['TR'][i])

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


for ticker in tickers:
    ohlc_ltf[ticker]["bar_num"] = np.where(ohlc_ltf[ticker]["Candle"]=='Boring Candle',1,0)
  

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
        
# =============================================================================
# data['diff_price_acc']=data['diff_price']
# for i in range(1,len(data["bar_num"])):
#     if data["bar_num"][i]>100 and data['diff_price'][i]>0:
#         data["diff_price_acc"][i]=data["diff_price_acc"][i]+data["diff_price_acc"][i-1]
# 
# =============================================================================
# =============================================================================
# data.reset_index(inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==11:
#         x=i-11
#         for i in range(i,x,-1):
#             data.drop(i,inplace=True)
#             print(i)
# data.reset_index(drop=True,inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==10:
#         x=i-10
#         for i in range(i,x,-1):
#             data.drop(i,inplace=True)
#             print(i)            
# data.reset_index(drop=True,inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==9:
#         x=i-9
#         for i in range(i,x,-1):
#             data.drop(i,inplace=True)
#             print(i)
# data.reset_index(drop=True,inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==8:
#         x=i-8
#         for i in range(i,x,-1):
#             data.drop(i,inplace=True)
#             print(i)
# data.reset_index(drop=True,inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==7:
#          x=i-7
#          for i in range(i,x,-1):
#              data.drop(i,inplace=True)
#              print(i)
# data.reset_index(drop=True,inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==6:     
#         x=i-6
#         for i in range(i,x,-1):
#             data.drop(i,inplace=True)
#             print(i)  
# data.reset_index(drop=True,inplace=True)
# for i in range(1,len(data["bar_num"])):
#     if data['bar_num'][i]==5:           
#         x=i-5
#         for i in range(i,x,-1):
#             data.drop(i,inplace=True)
#             print(i)              
#   
#     
# #data.reset_index(drop=True,inplace=True) 
#      
# 
#     
# 
# data['Candle_cat'].replace({'': None},inplace=True)
# data.dropna(subset=['diff_price','Candle_cat'],inplace=True)
# data.reset_index(inplace=True)
# =============================================================================

for ticker in tickers:
    ohlc_ltf[ticker].reset_index(drop=True,inplace=True)
    ohlc_ltf[ticker]['date']=''
    for i in range(len(ohlc_ltf[ticker]["Datetime"])):
        ohlc_ltf[ticker]['date'][i]=ohlc_ltf[ticker].Datetime[i].date()
        
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


# =============================================================================
#     elif (data['Candle_cat'][i]=='Legout Candle' and data['bar_num'][i-1]==400 and data['Candle_cat'][i-5]=='Legin Candle') and (data.date[i]==data.date[i-1]==data.date[i-5]):
#       data['signal'][i]='Zone'
#     elif (data['Candle_cat'][i]=='Legout Candle' and data['bar_num'][i-1]==300 and data['Candle_cat'][i-4]=='Legin Candle') and (data.date[i]==data.date[i-1]==data.date[i-4]):
#         data['signal'][i]='Zone'
#     elif (data['Candle_cat'][i]=='Legout Candle' and data['bar_num'][i-1]==200 and data['Candle_cat'][i-3]=='Legin Candle') and (data.date[i]==data.date[i-1]==data.date[i-3]):
#         data['signal'][i]='Zone'
# =============================================================================




##### HTF
endhtf = dt.datetime.today()
starthtf = endhtf-dt.timedelta(90)

ohlc_htf_copy={}
for ticker in tickers:
    temp=yf.download(ticker,starthtf,endhtf,interval=htf)
    temp.dropna(how='any',inplace=True)
    ohlc_htf_copy[ticker]=temp


def MACD(DF,a=12,b=26,c=9):
    df=DF.copy() # to avoid change in original dataframe
    df['ma_fast']=df['Adj Close'].ewm(span=a,min_periods=a).mean()
    df['ma_slow']=df['Adj Close'].ewm(span=b,min_periods=b).mean()
    df['macd']=df['ma_fast']-df['ma_slow']
    df['signal']=df['macd'].ewm(span=c,min_periods=c).mean()
    return (df['macd'],df['signal'])


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


ohlc_htf = copy.deepcopy(ohlc_htf_copy)

for ticker in tickers:
    ohlc_htf[ticker]["macd"]= MACD(ohlc_htf[ticker],12,26,9)[0]
    ohlc_htf[ticker]["macd_sig"]= MACD(ohlc_htf[ticker],12,26,9)[1]
    ohlc_htf[ticker]["macd_slope"] = slope(ohlc_htf[ticker]["macd"],5)
    ohlc_htf[ticker]["macd_sig_slope"] = slope(ohlc_htf[ticker]["macd_sig"],5)
    ohlc_htf[ticker].dropna(inplace=True)

for ticker in tickers:
    ohlc_htf[ticker].reset_index(inplace=True)
    ohlc_htf[ticker]['date']=''
    for i in range(len(ohlc_htf[ticker]["Date"])):
        ohlc_htf[ticker]['date'][i]=ohlc_htf[ticker].Date[i].date()
new_dict={}
for ticker in tickers:
    new_dict[ticker]=pd.merge(left=ohlc_ltf[ticker], right = ohlc_htf[ticker], on = 'date', how='inner')

for ticker in tickers:
    new_dict[ticker]['Real Zone']=''
    for i in range(len(new_dict[ticker])):
        if new_dict[ticker]['signal'][i]=='buy' and new_dict[ticker]['macd'][i]>new_dict[ticker]['macd_sig'][i] and new_dict[ticker]['macd_slope'][i]>new_dict[ticker]['macd_sig_slope'][i] and new_dict[ticker]['white area'][i]=='yes':
            new_dict[ticker]['Real Zone'][i]='buy'
        elif new_dict[ticker]['signal'][i]=='sell' and new_dict[ticker]['macd'][i]<new_dict[ticker]['macd_sig'][i] and new_dict[ticker]['macd_slope'][i]<new_dict[ticker]['macd_sig_slope'][i] and new_dict[ticker]['white area'][i]=='yes':
            new_dict[ticker]['Real Zone'][i]='sell'

for ticker in tickers:       
    new_dict[ticker]['Real Zone'].replace({'':np.NaN},inplace=True)

Final_ticker={}
for ticker in tickers:
    Final_ticker[ticker]=new_dict[ticker][new_dict[ticker]['date']==datestamp]
    Final_ticker[ticker]=Final_ticker[ticker].loc[Final_ticker[ticker]['Real Zone'].notnull(),['Datetime','Real Zone']]


Signal={}
for ticker in tickers:
    Signal[ticker]=ohlc_ltf[ticker][ohlc_ltf[ticker]['date']==datestamp]
    Signal[ticker]=Signal[ticker][['Datetime','signal']]
    Signal[ticker][Signal[ticker].signal.notnull()]

