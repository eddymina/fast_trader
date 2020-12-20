import numpy as np 
import requests
import pandas as pd 
import time 
import datetime as dt 

from .dates import Date,Dates
from .tensor_ops import TraderArray,TraderMatrix,TraderDict
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

        self.base_data,self.meta,self.datetimes,self.quote_data=self._data()

    def to_dict(self):
        res_dict= self.quote_data.copy()
        res_dict['date_info']=self.datetimes
        return res_dict

    def to_df(self):
        return pd.DataFrame(data= self.quote_data,index=self.datetimes).sort_index()

    def to_tDict(self):
        inds= np.argsort(self.datetimes)
        date_info=Dates(self.datetimes[inds])
        datum= {}
        for k,v in self.quote_data.items():
            datum[k]=TraderArray(np.array(v)[inds],date_info,name=k,sort_dates=False)
        return TraderDict(datum,name=self.ticker) 

    def to_tMat(self):
        inds= np.argsort(self.datetimes)
        date_info=Dates(self.datetimes[inds])
        data=np.column_stack(list(self.quote_data.values()))
        attrs=list(self.quote_data.keys())
        return TraderMatrix(data=data,date_info=date_info,
                            attrs=attrs,name=self.ticker)    

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
                start = today-dt.timedelta(days=59)
    
            elif today-start > dt.timedelta(days=60):
                raise TraderError('Start Date out of range. For {} data, max range is 60 days. Start Max= ({})'.format\
                                  (self.interval,today-dt.timedelta(days=59)))
            
            start= int(dt.datetime.timestamp(start))
                
        elif self.interval in ['60m','90m']:

            if start is None:
                start = today-dt.timedelta(days=729)

            elif today-start > dt.timedelta(days=730):
                raise TraderError('Start Date out of range. For {} data, max range is 730 days. Start Max= ({})'.format\
                                  (self.interval,today-dt.timedelta(days=729)))

            start= int(dt.datetime.timestamp(start))
                
        elif self.interval in ['1d','5d','1wk','1mo','3mo']:
            if start is None:
                start= dt.datetime(year=1900, month=1,day=1)

            start= int(dt.datetime.timestamp(start))

        else:
            raise TraderError("interval arguments must be in '1m','2m','5m','15m','60m','90m','1d','5d','1wk','1mo','3mo'")
            
        return start,end

    def _data(self):
        resp = requests.get(url=self.url, params=self.params)
        if resp.status_code != 200:
            raise TraderError('Invalid parameter set. Status Code:{}'.format(resp.status_code))

        data = resp.json()

        base_data=data['chart']['result'][0]
        quote_data= base_data["indicators"]["quote"][0]
        try:
            quote_data.update(base_data["indicators"]['adjclose'][0])
        except:
            pass
        datetimes=np.array(list(map(dt.datetime.fromtimestamp,base_data['timestamp'])),dtype='O')#pd.to_datetime(base_data['timestamp'],unit="s")

        return base_data,base_data['meta'],datetimes,quote_data





