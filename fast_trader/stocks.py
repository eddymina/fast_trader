import numpy as np 
import requests
import pandas as pd 
import time 
import datetime as dt 

from .dates import Date,Dates
from .array_ops import TraderDict,TraderArray
from .utils import TraderError, TraderWarning


def stock_name(symbol):
    """
    symbol:: stock symbol 
    returns full stock name 
    """
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    result = requests.get(url).json()
    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']

class YahooStock:
    
    def __init__(self,ticker,start=None,end=None,
                interval='1d'):
        """
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            
        start: str
            Download start date string (YYYY-MM-DD) or _datetime.
            Default is 1900-01-01
        end: str
            Download end date string (YYYY-MM-DD) or _datetime.
            Default is now
        """
        
        self.ticker = ticker 
        self.url = "https://query1.finance.yahoo.com/v8/finance/chart/{}".format(ticker)
        
        self.interval = interval.lower()
        start,end= self.handle_dates(start,end)
        
        params = {"period1": start, "period2": end}
        
        params["interval"] = self.interval
            
        self.params=params
            
    def handle_dates(self,start,end):
        today = dt.datetime.now()

        if start is not None:
            start= Date(start).date
            
        if end is None:
            end = int(dt.datetime.timestamp(today))
        else:
            end = int(dt.datetime.timestamp(Date(end).date))
            
        if self.interval=='1m':

            if start is None:
                start = today-dt.timedelta(days=6)

            elif today - start > dt.timedelta(days=7):
                raise TraderError('Start Date out of range. For {} data, max range is 7 days. Start Max= ({})'.format\
                                  (self.interval,today-dt.timedelta(days=6)))

            start= int(dt.datetime.timestamp(start))
                
        elif self.interval in ['2m','5m','15m']:

            if start is None:
                start = today.date-dt.timedelta(days=59)
    
            elif today-start > dt.timedelta(days=60):
                raise TraderError('Start Date out of range. For {} data, max range is 60 days. Start Max= ({})'.format\
                                  (self.interval,today.date-dt.timedelta(days=59)))
            
            start= int(dt.datetime.timestamp(start.date))
                
        elif self.interval in ['60m','90m']:

            if start is None:
                start = today-dt.timedelta(days=729)

            elif today-start > dt.timedelta(days=730):
                raise TraderError('Start Date out of range. For {} data, max range is 730 days. Start Max= ({})'.format\
                                  (self.interval,today.date-dt.timedelta(days=729)))

            start= int(dt.datetime.timestamp(start.date))
                
        elif self.interval in ['1d','5d','1wk','1mo','3mo']:
            if start is None:
                start= dt.datetime(year=1900, month=1,day=1)

            start= int(dt.datetime.timestamp(start))

        else:
            raise TraderError("interval arguments must be in '1m','2m','5m','15m','60m','90m','1d','5d','1wk','1mo','3mo'")
            
        return start,end

    def _data(self,adjusted):
        resp = requests.get(url=self.url, params=self.params, proxies=None)
        if resp.status_code != 200:
            raise TraderError('Invalid parameter set. Status Code:{}'.format(resp.status_code))
        if "Will be right back" in resp.text:
            raise TraderError("*** YAHOO! FINANCE IS CURRENTLY DOWN!")
        data = resp.json()
        self.data=data
    
        data_d= data['chart']['result'][0]["indicators"]["quote"][0]
        if adjusted:
            if "adjclose" in data['chart']['result'][0]["indicators"].keys():
                data_d.update({'adjusted':data['chart']['result'][0]["indicators"]['adjclose'][0]['adjclose']})
            else:
                TraderWarning('No adjustment data found')
  
        datetimes=pd.to_datetime(data['chart']['result'][0]['timestamp'],unit="s")-dt.timedelta(hours=4)

        return datetimes,data_d

        
    def history(self,adjusted=False):

    
        datetimes,data_d= self._data(adjusted=adjusted)
        inds= np.argsort(datetimes)
        date_info=Dates(datetimes[inds])
        datum= {}
        for k,v in data_d.items():
            v=np.array(v)
            datum[k]=TraderArray(v[inds],date_info,name=k,sort_dates=False)
        return TraderDict(datum,name=self.ticker)

# class YahooStock:
    
#     def __init__(self,ticker,start=None,end=None,
#                 interval='1d'):
#         """
#         interval : str
#             Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
#             Intraday data cannot extend last 60 days
            
#         start: str
#             Download start date string (YYYY-MM-DD) or _datetime.
#             Default is 1900-01-01
#         end: str
#             Download end date string (YYYY-MM-DD) or _datetime.
#             Default is now
#         """
        
#         self.ticker = ticker 
#         self.url = "https://query1.finance.yahoo.com/v8/finance/chart/{}".format(ticker)
        
#         self.interval = interval.lower()
#         start,end= self.handle_dates(start,end)
        
#         params = {"period1": start, "period2": end}
        
#         if self.interval == "30m":
#             params["interval"] = "15m"
#         else: 
#             params["interval"] = self.interval
            
#         self.params=params
            
#     def handle_dates(self,start,end):
#         today = Date(dt.datetime.now())

#         if start is None:
#             start = Date('01-01-1900')
#         else:
#             start = Date(start)
# #             start = int(datetime.timestamp(Date(start).datetime))
#         if end is None:
#             end = int(dt.datetime.timestamp(today.datetime))
#         else:
#             end = int(dt.datetime.timestamp(Date(end).datetime))
            
#         if self.interval=='1m':
#             if today.datetime-start.datetime > dt.timedelta(days=7):
#                 raise TraderError('Start Date out of range. For {} data, max range is 7 days. Start Max= ({})'.format\
#                                   (self.interval,today.datetime-dt.timedelta(days=6)))
#             else:
#                 start= int(dt.datetime.timestamp(start.datetime))
                
#         elif self.interval in ['2m','5m','15m','30m']:
#             if today.datetime-start.datetime > dt.timedelta(days=60):
#                 raise TraderError('Start Date out of range. For {} data, max range is 60 days. Start Max= ({})'.format\
#                                   (self.interval,today.datetime-dt.timedelta(days=59)))
#             else:
#                 start= int(dt.datetime.timestamp(start.datetime))
                
#         elif self.interval in ['60m','90m']:
#             if today.datetime-start.datetime > dt.timedelta(days=730):
#                 raise TraderError('Start Date out of range. For {} data, max range is 730 days. Start Max= ({})'.format\
#                                   (self.interval,today.datetime-dt.timedelta(days=729)))
#             else:
#                 start= int(dt.datetime.timestamp(start.datetime))
                
#         elif self.interval in ['1d','5d','1wk','1mo','3mo']:
#             start= int(dt.datetime.timestamp(start.datetime))

#         else:
#             raise TraderError("interval arguments must be in '1m','2m','5m','15m','30m',60m','90m','1d','5d','1wk','1mo','3mo'")
            
#         return start,end
        
#     def history(self,adjusted=False):
    
#         resp = requests.get(url=self.url, params=self.params, proxies=None)
#         if resp.status_code != 200:
#             raise TraderError('Invalid parameter set. Status Code:{}'.format(resp.status_code))
#         if "Will be right back" in resp.text:
#             raise TraderError("*** YAHOO! FINANCE IS CURRENTLY DOWN!")
#         data = resp.json()

#         df= pd.DataFrame(data['chart']['result'][0]["indicators"]["quote"][0],
#                             index=pd.to_datetime(data['chart']['result'][0]['timestamp'],unit="s"))
        
#         if adjusted:
#             if "adjclose" in data['chart']['result'][0]["indicators"].keys():
#                 df["adjclose"]= data['chart']['result'][0]["indicators"]['adjclose'][0]['adjclose']
#                 adjust= True
#             else:
#                 TraderWarning('No adjustment data found')
#                 adjust = False 
#         else:
#             adjust = False 
                
#         df.sort_index(inplace=True)
        
#         if self.interval == "30m":
#             quotes2 = df.resample('30T')
#             df = pd.DataFrame(index=quotes2.last().index, data={
#                 'open': quotes2['open'].first(),
#                 'high': quotes2['high'].max(),
#                 'low': quotes2['low'].min(),
#                 'close': quotes2['close'].last(),
#                 'volume': quotes2['volume'].sum()
#             })
#             if adjust is True:
#                 df['adjclose']=quotes2['adjclose'].last(),
                
#         df.dropna(inplace=True)
#         return df





