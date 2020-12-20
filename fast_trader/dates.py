"""
Date processing frame work. All dates are 
processing to global variable (DATE_FORMAT)
For different structure this var must be changed.

Its default is '%m-%d-%Y %H:%M:%S'
"""

import datetime as dt 
from pandas.core.tools.datetimes import _guess_datetime_format
from .utils import TraderError, TraderWarning 
import numpy as np 
import pandas as pd 

DATE_FORMAT="%m-%d-%Y %H:%M:%S"

def NYC_TIME():
    """
    Time in NYC
    """
    return (dt.datetime.now(dt.timezone.utc)-dt.timedelta(hours=4)).replace(tzinfo=None)

def year(yr):
    """
    Input string or day for year info
    """
    return dt.datetime(year=int(yr), month=1,day=1)

def str2date(date,dformat=DATE_FORMAT):
    """
    Convert a dateobject to string
    """
    return dt.datetime.strptime(date,dformat)

def date2str(date,dformat=DATE_FORMAT):
    """
    Convert a string object to date
    """
    return date.strftime(dformat)

def guess_datetimef(date):
    """
    Uses pandas inner tools to guess 
    what datetime format is for a date object
    is. (will be converted to string)
    """
    out= _guess_datetime_format(str(date))
    if out is None:
        raise TraderError('Unable to guess dateformat.')
    return out 

# def date_sorter(data,date_info):
#     """
#     Sort data and dateinformation from 
#     old to new. 
#     """
#     D = Dates(date_info)
#     inds = np.argsort(D.dates)
#     return TraderArray(data[inds], D.dates[inds])

# def nearest_date(array, value):
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

class Date:
    def __init__(self,date,dformat=None):
        """
        Date Object 
        ...................Inputs...................
        
        date:: (str),datetime_object (1 only)
        dformat:: dateformat to parse data. If dformat is None format will be guessed
        
        ...................Outputs...................
        
        Date object formated based on DATE FORMAT
        attr:: 
            .datestr (date as a string)
            .datetime (datetime object)
    
        ...................Example(s)...................  
        
        >> D= Date('5/10/2020')
        <trader Date:: 05-10-2020 00:00:00>
        
        >> Date('5/10/2020').datestr()
        '05-10-2020 00:00:00'
        
        >> Date('5/10/2020').datetime
        datetime.datetime(2020, 5, 10, 0, 0)
        
        >> Date(Date('5/10/2020')) #will pass if date items input 
        """

        self.trader_date= True
        if hasattr(date, 'trader_date') is True: #if its a Dates Class Pass Attr
            self.__dict__ = date.__dict__
            
        if type(date) in [str, np.str_]:
            if dformat is None:
                dformat=guess_datetimef(str(date)) #guessing format
            #if input is string, will try to convert to datetime
            self.date= str2date(date,dformat=dformat) #get datetime from guess 
        else: self.date= date#str2date(self.datestr)

    def datestr(self):
        try:
            datestr= date2str(self.date,dformat=DATE_FORMAT)
        except:
            raise TraderError('Not understandable date object.')
        return datestr

    def to_datetime(self):
        try:
            return str2date(self.datestr())
        except:
            return pd.to_datetime(self.date)


    def to_npdatetime64(self):
        try:
            return np.datetime64(self.date)
        except:
            try:
                return np.datetime64(self.to_datetime())
            except:
                raise TraderError('Unable to convert to format.')

    def __repr__(self):
        return '<trader Date:: {}>'.format(self.date)
    

class Dates:
    """
    Dates Object #storing a (1D) series 
    ...................Inputs...................

    date_info:: 1D arraylike input of (str),datetime_object(s)
    same_format:: dateformat to parse information (will be guessed). 

        If true, will assume that the first format is same as other inputs 
        (much faster)
    ...................Outputs...................

    Dates object formed based on DATE FORMAT
    attr::
        .dates (array of Date Items)
        .datestrings (dates as a string)
        .datetimes (datetimes array )
        .min (oldest datetime)
        .max (most recent datetime)

    .time_delta()
        returns::
        tdeltas (dates[1:] - dates[:-1] ) #will be positive if date info sorted from older to new
        tdelta  (unique time delta(s))

    ...................Example(s)...................  
    >> import pandas as pd 
    >> D= Dates('5/10/2020') #not recommended (use dates)
    [<trader Date:: 05-10-2020 00:00:00>]

    >> dates=Dates(pd.date_range(start='1/1/2018', end='1/08/2018'))
    <TraderDates:: shape=(8,), start=2018-01-01 00:00:00, stop=2018-01-08 00:00:00>

    >> dates[4:] #indexable
    <TraderDates:: shape=(4,), start=2018-01-05 00:00:00, stop=2018-01-08 00:00:00>

    >> for d in dates[4:]:
          print(d) #iterable 
        
        <trader Date:: 01-05-2018 00:00:00>
        <trader Date:: 01-06-2018 00:00:00>
        <trader Date:: 01-07-2018 00:00:00>
        <trader Date:: 01-08-2018 00:00:00>
        
    >> dates.dates #returns dates 
    >> dates.datetimes #returns datetimes 
    >> print(dates.time_delta()[0][0])
    1 day, 0:00:00
    """
    def __init__(self,date_info,same_format=True):
        
        if hasattr(date_info, 'trader_dates') is True: #if its a Dates Class Pass 
            self.__dict__ = date_info.__dict__

        else:
            if hasattr(date_info,'__len__') is False: date_info= np.atleast_1d(date_info)
            dtype= pd.api.types.infer_dtype(date_info)

            if np.ndim(date_info)>1:
                raise TraderError('Data Information must be 1Dimensional')

            self.trader_dates=True

            if dtype == 'string':
                if same_format is True:
                    dformat=guess_datetimef(date_info[0])
                    self.dates= np.array([Date(d,dformat).date for d in date_info],dtype='O') #get dates info
                else:
                    self.dates= pd.to_datetime(date_info)
            elif 'datetime' in dtype:
                self.dates=date_info#np.array(date_info,dtype='O')
            else:
                raise TraderError('Invalid Date Information')
                
            self.min= Date(self.dates[0]).to_datetime()
            self.max= Date(self.dates[-1]).to_datetime()
            self.shape=(len(self.dates),)

    def nearest_date(self, value,return_ind=False):
        """
        Find the nearest date in the list of dates.

        value:: datetime object
        """
        idx = (np.abs(self.dates - value)).argmin()
        if return_ind:
            return idx,self.dates[idx]
        return self.dates[idx]

    def __len__(self):
        return len(self.dates)
        
    def __getitem__(self, i):
        """
        Allows trader array indexing 
        """
        return Dates(self.dates[i])
    
    def __iter__(self):
        """
        Trader array is iterable
        """
        return iter(self.dates)
    
    def __repr__(self):
        return '<TraderDates:: shape={}, start={}, stop={}>'.format(self.shape,self.min,self.max)
    


def daysinmonth(month,year):
    """
    How many days are in certain 
    month and year. 
    
    month:: int [1,12]
    year:: 4 digit int (1997)
    """
    date= dt.date(year=year,month=month,day=1)
    res= (dt.date(year = date.year+int(date.month/12),
                  month = date.month % 12 +1, 
                  day = 1))
    return (res-date).days

def addyears(date,n=0):
    """
    Add or minus year(s) from a 
    datetime object. 
    
    date:: datetime obj
    n:: n year(s) to add/subtract (int)
    """
    adj_year=date.year+n
    day=min(date.day,daysinmonth(date.month,adj_year))
    return dt.datetime(year = adj_year,
                          month = date.month, 
                          day = day)

def addmonths(date,n=0):
    """
    Add or minus month(s) from a 
    datetime object. 
    
    date:: datetime obj]
    n:: n month(s) to add/subtract (int)
    """
    sm=(date.month+n)%12
    adj_year= date.year+divmod(date.month+n,12)[0]
    day= min(date.day,daysinmonth(sm,adj_year))
    return dt.datetime(year = adj_year, 
                      month =sm,
                      day = day)


class DateIndex:
    def __init__(self,sdates,stock_list):
        
        self.stock_list= stock_list
        
        self.sdates=sdates
        #flatten dates 
        self.sdatesf=[item for sublist in sdates for item in sublist]
        
        #unique sorted dates 
        self.date_index= np.array(sorted(set(self.sdatesf)),dtype='O') 
        self.ind_dict = dict((k,i) for i,k in enumerate(self.date_index))

        self.matrices=[]
        self.columns= []
        
    def to_matrix(self):
        for i,candidate_date in enumerate(self.sdates):
            
            datum=self.stock_list[i].data
            matrix= np.empty(shape=(len(self.date_index),datum.shape[1]),dtype='float32')
            matrix[:]=np.nan
            matrix[self._inter(candidate_date)] = datum
            self.matrices.append(matrix)

            self.columns.extend(tuple([(self.stock_list[i].name,attr) for attr in self.stock_list[i].attrs]))
            
        self.matrices= np.hstack(self.matrices)

    def _inter(self,candidate_dates):
        return [self.ind_dict[x] for x in candidate_dates]



class TimeDelta:
    """
    Extension of datetime timedelta
    """
    def __init__(self, 
                 seconds=0, 
                 microseconds=0, 
                 milliseconds=0, 
                 minutes=0, 
                 hours=0, 
                 days=0,
                 weeks=0,
                 months=0,
                 years=0):
        
        self.datetime_td=dt.timedelta(days, seconds, 
                                   microseconds, milliseconds,
                                   minutes, hours, weeks)
        
        self.months,self.years= months,years
        
    def __add__(self, other):
        return addmonths(addyears(other,self.years),self.months)+self.datetime_td

    def __radd__(self, other):
        return addmonths(addyears(other,self.years),self.months)+self.datetime_td
    
    def __sub__(self, other):
        return addmonths(addyears(other,-self.years),-self.months)-self.datetime_td
        
    def __rsub__(self, other):
        return addmonths(addyears(other,-self.years),-self.months)-self.datetime_td

