import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from math import sqrt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from .utils import TraderError,TraderWarning,isTrader
from .dates import DATE_FORMAT,Dates,Date,date2str,str2date,guess_datetimef

class TraderArray: 
    def __init__(self,data,date_info=None,name='tarray',sort_dates=False):
        """
        Array Class
        """
        self.trader_array=True
        self.data = np.atleast_1d(data).astype(np.float64)
        self.ndim= self.data.ndim
        if self.ndim>1: 
            raise TraderError('Data must be 1D shape')
        
        self.shape=(len(self.data),)
        self.length = self.size= len(self.data)
        self.name = str(name)
        self.min=self.data.min()
        self.max=self.data.max()

        if (self.min or self.max) == np.nan:
            TraderWarning('Nans detected. Please use dropna() method or remove before initializing Array.')
        
        self.date_assign(date_info)
        if self.date_info is not None:
            if len(self.date_info) != len(self.data):
                raise TraderError('Date information must be the same size.')
            if sort_dates:
                inds= np.argsort(self.date_info.dates)
                self.date_info = self.date_info[inds]
                self.data= self.data[inds]
        else:
            self.date_info = None

    
    def __len__(self):
        """
        Trader Array Length 
        """
        return self.length
            
    def __getitem__(self, i):
        """
        Allows trader array indexing 
        """
        assert np.ndim(i)<=1, 'indexing should be 1d'
        if self.date_info:
            return TraderArray(self.data[i],self.date_info[i],name=self.name)
        else:
            return TraderArray(self.data[i],None,name=self.name)
    
    def __repr__(self):
        """
        Representation of array. 
        """
        base= '<TraderArray:: shape={}, min=({:.2f}), max=({:.2f})'.format\
        (self.shape,self.min,self.max)
        if self.date_info is not None:
            base+=', range=({},{})'.format(date2str(self.date_info.min.date(),'%m-%d-%Y'),
                                              date2str(self.date_info.max.date(),'%m-%d-%Y'))
        if self.name: 
            base+=', name="{}"'.format(self.name)
        return base+'>'
    
    def __iter__(self):
        """
        Trader array is iterable
        """
        if self.date_info:
            for i in range(self.length):
                yield(self.data[i],self.date_info.dates[i])
        else:
            for i in self.data:
                yield i

    def __mul__(self, other):
        name='mult_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(self.data*other.data,date_info=di,name=name)
        else:
            return TraderArray(self.data*other,date_info=self.date_info,name=name)
    
    def __rmul__(self, other):
        name='rmult_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(self.data*other.data,date_info=di,name=name)
        else:
            return TraderArray(self.data*other,date_info=self.date_info,name=name)
    
    def __truediv__(self, other):
        name='div_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(self.data/other.data,date_info=di,name=name)
        else:
            return TraderArray(self.data/other,date_info=self.date_info,name=name)
    
    def __rtruediv__(self, other):
        name='rdiv_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(other.data/self.data,date_info=di,name=name)
        else:
            return TraderArray(other/self.data,date_info=self.date_info,name=name)
    
    def __add__(self, other):
        name='add_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(self.data+other.data,date_info=di,name=name)
        else:
            return TraderArray(self.data+other,date_info=self.date_info,name=name)

    def __radd__(self, other):
        name='radd_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(self.data+other.data,date_info=di,name=name)
        else:
            return TraderArray(self.data+other,date_info=self.date_info,name=name)
    
    def __sub__(self, other):
        name='sub_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(self.data-other.data,date_info=di,name=name)
        else:
            return TraderArray(self.data-other,date_info=self.date_info,name=name)
        
    def __rsub__(self, other):
        name='rsub_op'
        if isTrader(other).res:
            if np.array_equal(self.date_info.dates,other.date_info.dates):
                di = self.date_info
            else:
                di = None 
            return TraderArray(other.data-self.data,date_info=di,name=name)
        else:
            return TraderArray(other-self.data,date_info=self.date_info,name=name)
    
    def __pow__(self,other):
        return TraderArray(self.data**int(other),self.date_info,self.name)

    def __lt__(self,other):
        return  self.data<other

    def __le__(self,other):
        return self.data<=other

    def __gt__(self,other):
        return self.data>other

    def __ge__(self,other):
        return self.data>=other

    def __eq__(self,other):
        return self.data==other

    def sign(self):
        return TraderArray(np.sign(self.data),self.date_info)

    def query(self,q,return_inds=False):
        return self.where(eval(q.replace('data','self.data').replace('date_info','self.date_info.dates')),return_inds)

    def where(self,a,return_inds=False):
        inds=np.argwhere(a).flatten()
        if inds.size == 0:
            raise TraderError('No indices satisfy these restrictions.')
        if return_inds:
            return inds,self.__getitem__(inds)
        return self.__getitem__(inds)

    def greater_than(self,a):
        assert np.ndim(a) == 0, 'must be 0D int or float'
        return self.where(self.data>a)

    # def equal_to(self,a):
    #     assert np.ndim(a) == 0, 'must be 0D int or float'
    #     return self.where(self.data==a)

    def less_than(self,a):
        assert np.ndim(a) == 0, 'must be 0D int or float'
        return self.where(self.data<a)
            
    def mean(self):
        """
        Mean
        """
        return sum(self.data)/self.length

    def var(self):
        """
        Variance 
        """
        #DimensionError.dim_err(self.data)
        mu= self.mean()
        return sum([(data_point- mu)**2 for data_point in self.data])/self.length

    def dropna(self,inplace=None,newarray=False):
        if np.isnan(self.min):
            nan_inds= np.isnan(self.data)

            if inplace is not None:
                np.place(self.data,nan_inds,inplace)
                data= self.data
                date_info=self.date_info
            else:
                data= self.data[~nan_inds]
                if self.date_info is not None:
                    date_info=self.date_info[~nan_inds]
                else:
                    date_info = None

            if newarray:
                return TraderArray(data,date_info=date_info,
                                name=self.name,sort_dates=True)
            else:
                self.__init__(data,date_info=date_info,
                name=self.name,sort_dates=True)
        return None
    
    def std(self):
        """
        Standard Deviation"
        """
        return sqrt(self.var())
    
    def mvg_avg(self,window):
        """
        Moving Average 
        """
        ret = np.cumsum(self.data)
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1:] / n
        
        if self.date_info is not None:
            di = self.date_info
        else: 
            di= None 
        return TraderArray(ret,di,name='mvg_avg_{}'.format(window))

    def fourier_decomp(self,ncomps=21):
        fft = np.fft.fft(self.data)
        fft[ncomps:-ncomps]=0

        return TraderArray(np.abs(np.fft.ifft(fft)),self.date_info,name='fft_decomp_{}'.format(ncomps))

    def diff(self,n=1):
        def _diff(x):
            return np.diff(x,n)
        if self.date_info:
            di = self.date_info[:-n]
        else:
            di = None
        return self.np_func(_diff,di)

    def num_grad(selfcenter=True):
        """
        Illustration of Numpy Gradient 
        """
        if order<=0: 
            return self.data
        di= None
        y = self.data

        if center:
            for _ in range(order):
                deriv=[]
                for i in range(1,len(y)-1):
                    deriv.append(((y[i+1]-y[i-1])/2))
                y=deriv
            if self.date_info is not None:
                di=self.date_info[order:-order]
            return TraderArray(deriv,di,name='num_grad_order_{}'.format(order))

        for _ in range(order):
            deriv=[]
            for i in range(0,len(y)-1):
                deriv.append((y[i+1]-y[i]))
            y=deriv
            if self.date_info is not None:
                di=self.date_info[:-order]   
        return TraderArray(deriv,di,name='num_grad_order_{}'.format(order))

    def minima_maxima(self):
        return TraderDict({'mins':
            self.__getitem__((np.diff(np.sign(np.diff(self.data))) > 0).nonzero()[0] + 1),
            'maxes':
            self.__getitem__((np.diff(np.sign(np.diff(self.data))) < 0).nonzero()[0] + 1)},
            name='minima_maxima')


    def np_func(self,func,date_info='same'):
        if date_info is 'same':
            date_info=self.date_info

        return TraderArray(func(self.data),date_info,name=self.name)
    
    def _plotter(self,scatter,date_inf,kwargs):
        if self.length<=1:
            TraderWarning('Not enough data to plot.')
        if scatter:
            plt_func = plt.scatter
        else:
            plt_func = plt.plot

        if date_inf:
            if self.date_info is not None:
                plt_func(self.date_info.dates,self.data,**kwargs)
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.gcf().autofmt_xdate()
            else:
                plt_func(np.arange(self.length),self.data,**kwargs)
        else:
            plt_func(np.arange(self.length),self.data,**kwargs)

            
    def plot(self,date_inf=True,**params):
        self._plotter(False,date_inf,params)

    def scatter(self,date_inf=True,**params):
        self._plotter(True,date_inf,params)

    def mvg_avg(self,window):
        """
        Moving Average 
        """
        ret = np.cumsum(self.data) 
        ret[window:] = ret[window:] - ret[:-window] #0 3 | 3 end | start -3 | 3 end 
        ret = ret[window - 1:] / window

        if self.date_info is not None:
            di = self.date_info[window - 1:]
        else: 
            di= None 
        return TraderArray(ret,di,name='mvg_avg_{}'.format(window))

    def minmax_scale(self,minv=0,maxv=1):
        data=(maxv-minv)*(self.data-self.min)/(self.max-self.min)+minv
        return TraderArray(data,self.date_info,name='min_max_scaled')

    def standard_scale(self):
        data= (self.data-self.mean())/self.std()
        return TraderArray(data,self.date_info,name='standard_scaled')
 
    def date_assign(self,dates):
        try:
            self.date_info= Dates(dates)
        except:
            self.date_info = None
        
    def index_dates(self,date):
        if self.date_info is None:
            raise TraderError.__date_info__()

        adjusted_inds= []

        for d in np.atleast_1d(date):
            adjusted_inds.append(self.date_info.nearest_date(self._date_handler(d),return_ind=True)[0])
        # adjusted_inds= np.argwhere(self.date_info.dates ==date).flatten()
        return TraderArray(self.data[adjusted_inds],self.date_info[adjusted_inds],name=self.name)

    def date_range(self,start=None,end=None,nearest=False):
        if self.date_info is None:
            raise TraderError.__date_info__()
        if start is None: 
            start= self.date_info.min
        else: 
            start = self._date_handler(start)
        if end is None: 
            end = self.date_info.max
        else: 
            end = self._date_handler(end)
        if start > end:
            raise TraderError('Start date must be <= end date.')

        if nearest:
            start =self.date_info.nearest_date(start)
            end =self.date_info.nearest_date(end)

        return self.where((start<=self.date_info.dates) & (end >= self.date_info.dates))

    def _date_handler(self,date):
        try:
            date_p= Date(date)
        except:
            TraderError.__date_type__()

        if date_p.date < self.date_info.min or date_p.date > self.date_info.max:
            TraderWarning('One of dates out of range ({},{})'.format(self.date_info.min,self.date_info.max))

        return date_p.date

    def buy_hold(self,nshares,plot=False):

        bought = self.data[0]*nshares
        # bought.name='bought_at_{:.2f}'.format(self.data[0])
        sold = self.data[-1]*nshares
        # sold.name='sold_at_{:.2f}'.format(self.data[-1])
        earned = sold-bought
        # earned.name = 'profit'
        
        adjusted = TraderArray(nshares*self.data,self.date_info,name=self.name)#_{}__{}'.format(s.date(),e.date()))
        if plot:
            plt.figure(figsize=(10,2))
            plt.title('Bought @:${:.2f}/share --> Sold @:${:.2f}/share | Earned ${:.2f}'.format(self.data[0],self.data[-1],earned))
            adjusted[[0,-1]].scatter(color='red')
            adjusted[[0,-1]].plot(color='pink',linestyle=':')
            adjusted.plot(color='black')

        return adjusted

    def to_df(self):
        if self.date_info is not None:
            return pd.Series(self.data,index= self.date_info.dates,name=self.name).to_frame()
        else:
            return pd.Series(self.data,name=self.name).to_frame()

    def to_numpy(self):
        if self.date_info is not None:
            return self.data,self.date_info.dates
        else:
            return self.data

    def split_on(self,splitval):

        if isinstance(splitval,int):

            return self.__getitem__(slice(None, splitval, None)),\
            self.__getitem__(slice(splitval, None, None))
            
        elif isinstance(splitval,float):
            assert splitval < 1 and splitval > 0, 'float must be in [0,1] of ratio to keep.'
            return self.__getitem__(slice(None,int(splitval*self.size), None)),\
            self.__getitem__(slice(int(splitval*self.size), None,None))
        
        else:
            try:
                date= Date(splitval).date
            except:
                raise TraderError('Invalid split on argument. Must be datetime,int,float')

            return self.date_range(None,date,nearest=True),self.date_range(date,None,nearest=True)

class TraderDict(dict):
    def __init__(self, dic,name=None):

        core_dict= dic.copy()
        if type(core_dict) is not dict:
            raise TraderError('Set Input As a Dictionary')

        self.__dict__ = core_dict
        self.tensor_dict=True

        self._core_dict= self._convert_keys_to_string(dic.copy())

        self.trader_dict = True
        self.nitems=self.ndim=len(self._core_dict)
        self.tensor=list(self._core_dict.values())

        isTrader(self.tensor).error()
        
        self._date = True
        for arr in self.tensor: 
            if arr.date_info is None:
                self._date = False
                break 

        self.shape= (self.nitems,[arr.shape[0] for arr in self.tensor])
        self.name=name
        self.attrs= list(self._core_dict.keys())

    def append(self,newd):

        self.__dict__.update(newd)

        d2str=self._convert_keys_to_string(newd)

        self._core_dict.update(d2str)
        self.nitems+=len(newd)
        self.ndim+=len(newd)

        ntensor=list(d2str.values())

        isTrader(ntensor).error()

        self.tensor.extend(ntensor)

        self._date = True
        for arr in ntensor: 
            if arr.date_info is None:
                self._date = False
                break 

        self.shape[1].extend([arr.shape[0] for arr in ntensor])
        self.attrs.extend(list(d2str.keys()))

    def __repr__(self):
        """
        Representation of array. 
        """
        base= '<TraderDict:: shape=({},{}),'.format(self.shape[0],self.shape[1])
        if self.nitems>5:
            base+=' attrs={}...,'.format(self.attrs[:4])
        else:
            base+=' attrs={},'.format(self.attrs)

        base+=' name={}'.format(self.name)

        return base+'>'
    
    def __len__(self):
        return self.nitems
    
    def __iter__(self):
        """
        Trader array is iterable
        """
        for i in range(self.nitems):
            yield(self.attrs[i],self.tensor[i])

    def index_attrs(self,i):
        type_i =type(i)
        if np.ndim(i) > 1:
            raise TraderError('Only 1D indexing supported. Use .to_numpy() or to_df() for index broadcasting.')
        if hasattr(i,'__len__'):
            if type(i[0]) is str:
                return TraderDict(dict([(a,self.__dict__[a]) for a in i]),
                                  name=self.name)
            return TraderDict(dict([(self.attrs[a],self.tensor[a]) for a in i])
                                       ,name=self.name)
        elif type_i is str:
            return self.__dict__[i]
        elif type_i is slice:
            return TraderDict(dict(zip(self.attrs[i],self.tensor[i])))
        elif type_i is int:
            return TraderDict(dict(zip([self.attrs[i]],[self.tensor[i]])),name=self.name)
        else:
            raise TraderError('Invalid indexing operation.')

    def _get_x(self,i):
        f='__getitem__'
        return TraderDict(self._arrayf(f,i=i),name=self.name)

    def __getitem__(self,i):
        f='__getitem__'

        if isinstance(i, tuple):
            assert len(i)==2,'only support for 2D multiindex'
            i1,i2= i 

            if i1 and i2 is not None:
                if isinstance(i1, slice) and type(i1.start) != int:
                    s= self.date_range(start=i1.start,end=i1.stop)
                else:
                    s= self._get_x(i1)

                return s.index_attrs(i2)

            if i1 is not None:
                if isinstance(i1, slice) and type(i1.start) != int:
                    return self.date_range(start=i1.start,end=i1.stop)
                return self._get_x(i1)

            if i2 is not None:
                return self.index_attrs(i2)

            if type(i.start) != int:
                return self.date_range(start=i.start,end=i.stop)
            return self._get_x(i1)

        if isinstance(i, list) or isinstance(i, int):
            return self._get_x(i)
        raise TraderError('Invalid Index(s)')

    def to_dict(self):
        return self._core_dict
            
    def to_numpy(self):

        def fill_nan(arr,n,nan_type):
            return np.concatenate([arr,np.repeat(nan_type,n-arr.shape[0])])
    
        date_info=[]
        arrays=[]
        max_size= max(self.shape[1])
        for arr in self.tensor:
            if self._date:
                date_info.append(fill_nan(np.array(list(arr.date_info.dates),dtype=np.datetime64),
                                      max_size,nan_type=np.datetime64("NaT")))

            arrays.append(fill_nan(arr.data,max_size,nan_type=np.nan))

        if self._date: 
            dates= np.array(date_info,dtype=np.datetime64)
            uniq=np.unique(dates)

            if uniq.shape[0]==dates.shape[1]:
                dates=uniq
                
            return {'date_tensor':dates,
                        'data_tensor':np.array(arrays,dtype=np.float64)}

        return {'data_tensor':np.array(arrays,dtype=np.float64)}
    
    def _convert_keys_to_string(self,dictionary):
        return dict((str(k).lower(), v) \
                for k, v in dictionary.items())
        
    def to_df(self):
        np_data =self.to_numpy()
        index= None
        if self._date and np_data['date_tensor'].ndim==1:
            index= np_data['date_tensor']

        dict_input = dict(zip(self.attrs,np_data['data_tensor']))
        return pd.DataFrame(dict_input,index=index)
    
    def _arrayf(self,f,*args,**kwargs):
        output_dict={}
        for i,name in enumerate(self.attrs):
            e='output_dict[name] = self.tensor[i]'+'.'+f+'(*args,**kwargs)'
            exec(e)
        return output_dict
        
    def fourier_decomp(self,*args,**kwargs):
        f='fourier_decomp'
        return TraderDict(self._arrayf(f,**kwargs),name=f)
    
    def mvg_avg(self,*args,**kwargs):
        f='mvg_avg'
        return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
    
    def num_grad(self,*args,**kwargs):
        f='num_grad'
        return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
        
    def np_func(self,*args,**kwargs):
        f='np_func'
        return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
    
    def minmax_scale(self,*args,**kwargs):
        f='minmax_scale'
        return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
        
    def standard_scale(self):
        f='standard_scale'
        return TraderDict(self._arrayf(f),name=f)
        
    def buy_hold(self,*args,**kwargs):
        f='buy_hold'
        return self._arrayf(f,*args,**kwargs)
        
    def dropna(self,inplace=None):
        output_dict={}
        for i in range(self.ndim):
            self.tensor[i]
            out = self.tensor[i].dropna(inplace=inplace,newarray=True)
            if out is None:
                output_dict[self.attrs[i]] = self.tensor[i]
            else:
                output_dict[self.attrs[i]] = out 
        self.__init__(output_dict,name=self.name)
        
    def plot(self,*args,**kwargs):
        f='plot'
        kwargs.update({'date_inf':self._date})
        self._arrayf(f,*args,**kwargs)
        
    def scatter(self,*args,**kwargs):
        f='scatter'
        kwargs.update({'date_inf':self._date})
        self._arrayf(f,*args,**kwargs)
    
    def date_range(self,*args,**kwargs):
        f='date_range'
        return TraderDict(self._arrayf(f,*args,**kwargs),name=f)

    def index_dates(self,*args,**kwargs):
        f='index_dates'
        return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
    
    
    

# def trader_stack(A,B):
#     """
#     1D stacking of trader array
#     """
#     if isTrader(A) is False or isTrader(B) is False:
#         raise TraderError('Item must be type TraderArray.')
#     data= np.concatenate([A.data,B.data])
#     if A.date_info and B.date_info is not None:
#         date_info=Dates(np.concatenate([A.date_info.dateti,B.date_info.datetimes]))
#     else:
#         date_info = None
#     return TraderArray(data,date_info)


def from_pd(s):
    type_v=type(s)
    if type_v==pd.core.frame.DataFrame:
        if s.shape[1] > 1:
            dic= {}
            for k, v in s.items():
                dic[k]= TraderArray(v.values,v.index,name=k)
            return TraderDict(dic,name='dataframe')
        else:
            return TraderArray(s.values,s.index,name=s.name)
    elif type_v == pd.core.series.Series:
        return TraderArray(s.values,s.index,name=s.name)
    else:
        raise TraderError('Type {} invalid.'.format(type_v))




