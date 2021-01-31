import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from math import sqrt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from operator import itemgetter 

from ..utils import TraderError,TraderWarning
from ..dates import DATE_FORMAT,Dates,Date,date2str,str2date,guess_datetimef,DateIndex


class _traderTensor: 
    def __init__(self,data,date_info,name,sort_dates):
        """
        base Tensor Class
        """
        self._data_to_array(data=data)

        self.ndim= self.data.ndim
        self._data_array_shape(data)
        
        self.shape= self.data.shape
        self.name = str(name)
        
        self._return_tensor_type= {'trader_array':TraderArray,
                                 'trader_matrix':self._return_tmat_attrs_same}.get(self.type)
        
        self.date_assign(date_info,sort_dates)
            
    def date_assign(self,dates,sort_dates=False):
        self.date_info= Dates(dates)
        if len(self.date_info) != len(self.data):
                raise TraderError('Date information must be the same size.')
        if sort_dates:
            inds= np.argsort(self.date_info.dates)
            self.date_info = self.date_info[inds]
            self.data= self.data[inds]
            
    def __len__(self):
        """
        Trader Matrix Length 
        """
        return self.shape[0]  
    
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
    
    def __mul__(self, other):
        name='mult_op'
        return self._math_op(func=np.multiply,A=self.data,B=other,name= name)
    
    def __rmul__(self, other):
        name='rmult_op'
        return self._rmath_op(func=np.multiply,A=self.data,B=other,name= name)
    
    def __truediv__(self, other):
        name='div_op'
        return self._math_op(func=np.divide,A=self.data,B=other,name= name)
    
    def __rtruediv__(self, other):
        name='rdiv_op'
        return self._rmath_op(func=np.divide,A=self.data,B=other,name= name)
    
    def __add__(self, other):
        name='add_op'
        return self._math_op(func=np.add,A=self.data,B=other,name= name)

    def __radd__(self, other):
        name='radd_op'
        return self._rmath_op(func=np.add,A=self.data,B=other,name= name)
    
    def __sub__(self, other):
        name='sub_op'
        return self._math_op(func=np.subtract,A=self.data,B=other,name= name)
        
    def __rsub__(self, other):
        name='rsub_op'
        return self._rmath_op(func=np.subtract,A=self.data,B=other,name= name)
    

    def index_dates(self,date):
        adjusted_inds= []
        for d in self._atleast_1d(date):
            adjusted_inds.append(self.date_info.nearest_date(self._date_handler(d),return_ind=True)[0])
        return self.__getitem__(adjusted_inds)


    def _date_handler(self,date):
        
        date_p= Date(date)
        if date_p.date < self.date_info.min or date_p.date > self.date_info.max:
            TraderWarning('One of dates out of range ({},{})'.format(self.date_info.min,self.date_info.max))
        return date_p.date
    
    def plot(self,**params):
        self._plotter(False,params)

    def scatter(self,**params):
        self._plotter(True,params)
        
    def split_on(self,splitval):
        if isinstance(splitval,int):

            return self.__getitem__(slice(None, splitval, None)),\
            self.__getitem__(slice(splitval, None, None))
            
        elif isinstance(splitval,float):
            assert splitval < 1 and splitval > 0, 'float must be in [0,1] of ratio to keep.'
            return self.__getitem__(slice(None,int(splitval*self.shape[0]), None)),\
            self.__getitem__(slice(int(splitval*self.shape[0]), None,None))
        
        else:
            try:
                date= Date(splitval).date
            except:
                raise TraderError('Invalid split on argument. Must be datetime,int,float')
            return self.date_range(None,date,nearest=True),self.date_range(date,None,nearest=True)
        
    def date_range(self,start=None,end=None,nearest=False):
        
        s = self.date_info.min
        e = self.date_info.max
        
        if start is not None: 
            s = self._date_handler(start)
        if end is not None: 
            e = self._date_handler(end)
        
        if s > e:
            raise TraderError('Start date must be <= end date.')
            
        if nearest:
            start_ind,_ =self.date_info.nearest_date(s,return_ind=True)
            end_ind,_ = self.date_info.nearest_date(e,return_ind=True)
            
            return self.__getitem__(slice(start_ind,end_ind))
        
        return self.where((s<=self.date_info.dates) & (e >= self.date_info.dates))
    
    def _atleast_1d(self,data):
        if data is None:
            return None
        if hasattr(data,'__len__') is False:
            return [data]
        return data
    
    def where(self,a,return_inds=False):
        inds=np.argwhere(a).squeeze()
        if inds.size == 0:
            raise TraderError('No indices satisfy these restrictions.')
        if return_inds:
            return inds,self.__getitem__(inds)
        return self.__getitem__(inds)
    
    def mean(self,*args,**kwargs):
        return self.data.mean(*args,**kwargs)
    
    def sum(self,*args,**kwargs):
        return self.data.sum(*args,**kwargs)
    
    def var(self,*args,**kwargs):
        return self.data.var(*args,**kwargs)
    
    def std(self,*args,**kwargs):
        return self.data.std(*args,**kwargs)
    
    def min(self,*args,**kwargs):
        return self.data.min(*args,**kwargs)
    
    def max(self,*args,**kwargs):
        return self.data.max(*args,**kwargs)
    
    def _return_tmat_attrs_same(self,data,date_info,name,sort_dates=False):
        return TraderMatrix(data=data,date_info=date_info,attrs=self.attrs,name=name,sort_dates=sort_dates)
    
    def _return_tensor(self,*targs,**tkwargs):
        return self._return_tensor_type(*targs,**tkwargs)
    
    def diff(self):
        return self.np_func(np.diff,self.date_info[:-1],axis=0)
    
    def mvg_avg(self,window):
        """
        Moving Average 
        """
        n=window
        ret = np.cumsum(self.data,axis=0)
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1:] / n
        
        return self._return_tensor(data=ret,
                            date_info= self.date_info[window-1:],
                            name='mvg_avg_{}'.format(window))
    
    
    def fft_decomp(self,ncomps=21):
        fft = np.fft.fft(self.data,axis=0)
        fft[ncomps:-ncomps]=0
        return self._return_tensor(data=np.abs(np.fft.ifft(fft,axis=0)),
                            date_info= self.date_info,
                            name='fft_decomp_{}'.format(ncomps))
    
    def sign(self):
        return self.np_func(func=np.sign,date_info=self.date_info)
    
    
    def minima_maxima(self):
        return {'mins':
            self.__getitem__((np.diff(np.sign(np.diff(self.data,axis=0)),axis=0) > 0).nonzero()[0] + 1),
            'maxes':
            self.__getitem__((np.diff(np.sign(np.diff(self.data,axis=0)),axis=0)  < 0).nonzero()[0] + 1)}
    
    
    def standard_scale(self):
        data= (self.data-self.mean(axis=0))/self.std(axis=0)
        return self._return_tensor(data,self.date_info,name='standard_scaled')
    
    
    def np_func(self,func,date_info='same',*fargs,**fkwargs):
        if date_info is 'same':
            date_info=self.date_info
            
        data=func(self.data,*fargs,**fkwargs)
        
        try:
            return self._return_tensor(data= data,
                            date_info= date_info,
                            name=self.name)
        
        except Exception as e :
            TraderWarning('Unable to convert to TraderObject. \nError: {}'.format(e))
            return date_info,data
        
        
    def copy(self):
        return self._return_tensor(data=self.data,date_info=self.date_info,
                            name=self.name)
    
    def isnan(self,where=False):
        if where:
            return np.isnan(self.min()),np.isnan(self.data)
        else:
            return np.isnan(self.min())

    def dropna(self,inplace=None,newarray=True):
        if self.isnan():
            kwargs_init={}
            nan_inds = np.isnan(self.data)
            
            if inplace is not None:
                np.place(self.data.copy(),nan_inds,inplace)
                
                data= self.data
                date_info=self.date_info
                
            else:
                if self.type == 'trader_matrix':
                    nan_inds=nan_inds.any(axis=-1)
                    kwargs_init.update(attrs=self.attrs)

                data= self.data[~nan_inds]
                date_info=self.date_info[~nan_inds]
            
            
            kwargs_init.update(dict(data=data,date_info=date_info,
                          name=self.name,sort_dates=True))
            
            kwargs_tensor=kwargs_init.copy()
            kwargs_tensor.pop('attrs',None)
    
        if newarray:
            return self._return_tensor(**kwargs_tensor
                                      )
        else:
            self.__init__(**kwargs_init)
            
    def __iter__(self):
        """
        Trader array is iterable
        """
        for i in range(self.shape[0]):
            yield np.hstack([self.date_info.dates[i],self.data[i]])

    def interactive(self,**kwargs):
        from ipywidgets import interact
        @interact(sindex=(0,self.date_info.shape[0]-1,1),eindex=(0,self.date_info.shape[0]-1,1))
        def _interactive(sindex=0,eindex=self.date_info.shape[0]-1):
            assert sindex < eindex,'start must be less than than end'
            plt.figure(figsize=(14,7))
            s,e=self.date_info.dates[sindex],self.date_info.dates[eindex]
            plt.title('Date Range: {} --> {}'.format(s,e))
            self.date_range(s,e).plot(**kwargs)

        _interactive()

class TraderArray(_traderTensor): 
    def __init__(self,data,date_info,name='tarray',sort_dates=False):
        """
        Array Class
        """
        super(TraderArray, self).__init__(data,date_info,name,sort_dates) 
        
    def _data_to_array(self,data):
        self.type='trader_array'
        self.data = np.array(self._atleast_1d(data),dtype=np.float32).squeeze()
        
    def _data_array_shape(self,data): 
        if self.ndim>1: 
            raise TraderError('Data must be 1D shape')
            
    def _math_op(self,func,A,B,name):

        if hasattr(B, 'date_info'):
            istrader_mat=B.type=='trader_matrix'
            dinfo = B.date_info.dates
            B= B.data
            if np.array_equal(self.date_info.dates,dinfo):
                if istrader_mat:
                    #tarray x tMatrix
                    return TraderMatrix(data= func(A,B),
                                date_info=self.date_info,
                                attrs= B.attrs,
                                name= name)
                
            else:
                raise TraderError('Dates Do not match')
                
        #tarray x tarray or tarray * array or tarray*scalar
        return TraderArray(data= func(A,B),
                                date_info=self.date_info,
                                name=name)
    
    def _rmath_op(self,func,A,B,name):
        
        if hasattr(B, 'date_info'):
            istrader_mat=B.type=='trader_matrix'
            dinfo = B.date_info.dates
            B= B.data
            if np.array_equal(self.date_info.dates,dinfo):
                if istrader_mat:
                    #tarray x tMatrix
                    return TraderMatrix(data= func(B,A),
                                date_info=self.date_info,
                                attrs= B.attrs,
                                name= name)
                
            else:
                raise TraderError('Dates Do not match')
                
        #tarray x tarray or tarray * array or tarray*scalar
        return TraderArray(data= func(B,A),
                                date_info=self.date_info,
                                name=name)
   
    def __repr__(self):
        """
        Representation of array. 
        """
        base= '<TraderArray:: shape={}'.format\
        (self.shape)

        base+=', date_range=({},{})'.format(date2str(self.date_info.min,'%m-%d-%Y'),
                                              date2str(self.date_info.max,'%m-%d-%Y'))
        if self.name: 
            base+=', name="{}"'.format(self.name)
        return base+'>'
    
    def __getitem__(self, i):
        """
        Allows trader array indexing 
        """
        assert np.ndim(i)<=1, 'indexing should be 1d'
        return TraderArray(self.data[i],self.date_info[i],name=self.name)
    
    def to_df(self):
        return pd.DataFrame(self.data,index= self.date_info.dates,columns=[self.name])
        
    def _plotter(self,scatter,kwargs):
        
        if self.shape[0]<=1:
            TraderWarning('Not enough data to plot.')

        if scatter:
            plt.scatter(self.date_info.dates,self.data,**kwargs)
        else:
            plt.plot(self.date_info.dates,self.data,**kwargs)

        plt.gcf().autofmt_xdate()
                
    def minmax_scale(self,minv=0,maxv=1):
        dmin,dmax= self.min(axis=0),self.max(axis=0)
        data=(maxv-minv)*(self.data-dmin)/(dmax-dmin)+minv
        return TraderArray(data,self.date_info,name='min_max_scaled')
    
    def buy_hold(self,nshares,plot=False):

        bought = self.data[0]*nshares
        # bought.name='bought_at_{:.2f}'.format(self.data[0])
        sold = self.data[-1]*nshares
        # sold.name='sold_at_{:.2f}'.format(self.data[-1])
        earned = sold-bought
        # earned.name = 'profit'
        
        adjusted = TraderArray(nshares*self.data,self.date_info,name=self.name)#_{}__{}'.format(s.date(),e.date()))

        if plot:
            plt.figure(figsize=(15,5))
            plt.title('Bought @:${:.2f}/share --> Sold @:${:.2f}/share | Earned ${:.2f}'.format(self.data[0],self.data[-1],earned))
            adjusted[[0,-1]].scatter(color='red')
            adjusted[[0,-1]].plot(color='pink',linestyle=':')
            adjusted.plot(color='black')

        return adjusted


class TraderMatrix(_traderTensor):
    def __init__(self,data,date_info,attrs=None,name='tTensor',sort_dates=False):
        """
        Matrix Class
        """
        super(TraderMatrix, self).__init__(data,date_info,name,sort_dates)   
        self._validate_attrs(attrs)
        
        for i,s in enumerate(self._split()):
            self.__dict__.update({self.attrs[i]:s})
        
    def _data_to_array(self,data):
        self.trader_tensor=True
        self.type='trader_matrix'
        self.data= np.array(data,dtype=np.float32)
        
    def _data_array_shape(self,data): 
        if self.ndim == 1:
            self.data= np.expand_dims(self.data,axis=-1)
            self.ndim=2
        elif self.ndim>2: 
            raise TraderError('Data must be 2D shape not {}'.format(self.shape))
            
    def _validate_attrs(self,attrs):
        if attrs is None:
            attrs=list(range(self.data.shape[1]))
        self.attrs=np.array(attrs,dtype='str').flatten()
        self.nitems= self.attrs.shape[0]
        if self.nitems != self.data.shape[1]:
            raise TraderError('Number of items {} must be same as number of cols {}'.\
                              format(self.nitems,self.data.shape[1]))
            
    def _math_op(self,func,A,B,name):

        if hasattr(B, 'date_info'):
            if np.array_equal(self.date_info.dates,B.date_info.dates):
                #tmatrix x tarray or tmatrix x tmatrix
                B= B.data 
            else:
                raise TraderError('Dates Do not match')

        #tmatrix * array or tarray*scalar
        try:
            return TraderMatrix(data= func(A,B),
                    date_info=self.date_info,
                    attrs= self.attrs,
                    name= name)
        
        except:
            return TraderMatrix(data= func(A,B),
                    date_info=self.date_info,
                    attrs= None,
                    name= name)
    
    def _rmath_op(self,func,A,B,name):
        
        if hasattr(B, 'date_info'):
            if np.array_equal(self.date_info.dates,B.date_info.dates):
                #tmatrix x tarray or tmatrix x tmatrix
                B= B.data 
            else:
                raise TraderError('Dates Do not match')

        #tmatrix * array or tarray*scalar
        try:
            return TraderMatrix(data= func(B,A),
                    date_info=self.date_info,
                    attrs= self.attrs,
                    name= name)

        except:
            return TraderMatrix(data= func(B,A),
                    date_info=self.date_info,
                    attrs= None,
                    name= name)

    def __repr__(self):
        """
        Representation of array. 
        """
        base= '<TraderMatrix:: shape={},'.format(self.shape)
        if self.date_info is not None:
            base+=' date_range=({},{})'.format(date2str(self.date_info.min,'%m-%d-%Y'),
                                              date2str(self.date_info.max,'%m-%d-%Y'))
        if self.nitems>4:
            base+=', attrs={}...,'.format(self.attrs[:3])
        else:
            base+=', attrs={},'.format(self.attrs)
            
        base+=' name="{}"'.format(self.name)
        return base+'>'
    
    def _attr2col_index(self,attr):
        return self.attrs.tolist().index(str(attr))
        
    def __getitem__(self, i):
        """
        Allows trader matrix indexing 
        """
        if type(i)==str:
            return self.__dict__[i]
        row_index,col_index=i,slice(None, None, None)
        if isinstance(i,tuple):
            row_index,col_index=i
            if isinstance(col_index,str):
                col_index= self._attr2col_index(col_index)
            elif hasattr(col_index,'__len__'):
                col_index= [self._attr2col_index(c) if isinstance(c,str) else c for c in col_index]
        
        col_index=col_index if isinstance(col_index,slice) else self._atleast_1d(col_index)
        row_index=row_index if isinstance(row_index,slice) else self._atleast_1d(row_index)

        try:
            return TraderMatrix(data=self.data[(row_index,col_index)],
                            date_info=self.date_info[row_index],
                            attrs=self.attrs[col_index],
                            name=self.name)
        except:
            raise TraderError('Invalid Index Params ({},{})'.format(row_index,col_index))
            
    def _plotter(self,scatter,kwargs):
        
        if self.shape[0]<=1:
            TraderWarning('Not enough data to plot.')
    
        if '_expand_dates' not in self.__dict__.keys():
            self._expand_dates= np.tile(self.date_info.dates,(self.nitems,1)).T
    
        if scatter:
            plt.scatter(self._expand_dates,self.data,**kwargs)
        else:
            plt.plot(self._expand_dates,self.data,**kwargs)

        plt.gcf().autofmt_xdate()
        

    def minmax_scale(self,minv=0,maxv=1):
        minv*=np.ones(self.nitems,np.float32)
        maxv*=np.ones(self.nitems,np.float32)
        dmin,dmax= self.min(axis=0),self.max(axis=0)
        data=(maxv-minv)*(self.data-dmin)/(dmax-dmin)+minv
        return TraderMatrix(data=data,date_info=self.date_info,attrs=self.attrs,name='min_max_scaled')
    
    def to_df(self):
        return pd.DataFrame(data=self.data,
                            index=self.date_info.dates,
                            columns=self.attrs)
    
    def tarray(self,attr):
        return TraderArray(data= self.data[:,self._attr2col_index(attr) if isinstance(attr,str) else attr],
                           date_info=self.date_info,
                           name=str(attr))

    def squeeze(self):
        if self.shape[1]==1:
            return self.tarray(self.attrs[0])
    
    def _split(self):
        tarrays=[]
        datasplit= np.hsplit(self.data,self.nitems)
        for i,attr in enumerate(self.attrs):

            tarrays.append(TraderArray(data= datasplit[i],
                           date_info=self.date_info,
                           name=str(attr)))
        return tarrays
    
class TraderDict:
    def __init__(self,tarrays,attrs=None,name=None):
        
        tarrays= self._atleast_1d(tarrays)
        
        if attrs is None: 
            attrs= [arr.name for arr in tarrays]
            
        self.attrs=np.array(attrs,dtype='str').flatten()
        
        for i,arr in enumerate(tarrays): 
            type_arr=type(arr)
            if type_arr == TraderArray:
                tarrays[i]= TraderMatrix(arr.data,arr.date_info,attrs=[self.attrs[i]],name=arr.name,sort_dates=False)
                
            elif type_arr!= TraderMatrix:
                raise TraderError('Must be array-like grouping of trader array not {}')
        
        self.type= 'trader_dict'
        self.nitems = len(tarrays)

        assert self.nitems==self.attrs.shape[0],\
        'number of attrs ({}) must be the same length as number of items ({})'.format(self.attrs.shape[0],self.nitems,)
        
        self.tensor=tarrays
        self.name=name
        
        lens=[]
        for i,arr in enumerate(tarrays):
            self.__dict__.update({self.attrs[i]:arr})
            lens.append(arr.shape[0])
            
        self.shape= (self.nitems,lens)
        
    def _atleast_1d(self,data):
        if hasattr(data,'__len__') is False:
            return [data]
        return data
    
    def __repr__(self):
        """
        Representation of array. 
        """
        shape1= str(self.shape[1][:4])+'...' if len(self.shape[1])>5 else self.shape[1]
        base= '<TraderDict:: shape=({},{}),'.format(self.shape[0],shape1)
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
            
        
    def fft_decomp(self,*fargs,**fkwargs):
        return TraderDict([arr.fft_decomp(*fargs,**fkwargs) for arr in self.tensor],
                          attrs=self.attrs,name='fourier_decomp')
    
    
    def plot(self,*fargs,**fkwargs):
        for arr in self.tensor:
            arr.plot(*fargs,**fkwargs) 
        
    def scatter(self,*fargs,**fkwargs):
        for arr in self.tensor:
            arr.scatter(*fargs,**fkwargs) 
    
    def date_range(self,*fargs,**fkwargs):
        return TraderDict([arr.date_range(*fargs,**fkwargs) for arr in self.tensor],
                          attrs=self.attrs,name=self.name)
    
    def index_dates(self,*fargs,**fkwargs):
        return TraderDict([arr.index_dates(*fargs,**fkwargs) for arr in self.tensor],
                          attrs=self.attrs,name=self.name)
    
    
    def minmax_scale(self,*fargs,**fkwargs):
        return TraderDict([arr.minmax_scale(*fargs,**fkwargs) for arr in self.tensor],
                          attrs=self.attrs,name=self.name)
    
    
    def _attr2col_index(self,attr):
        return self.attrs.tolist().index(str(attr))

    def __getitem__(self, i):
        """
        Allows trader matrix indexing 
        """
        if type(i)==str:
            return self.tmatrix(i)

        row_index,col_index=i,slice(None, None, None)
        if isinstance(i,tuple):
            row_index,col_index=i
            if isinstance(col_index,str):
                col_index= self._attr2col_index(col_index)
            elif hasattr(col_index,'__len__'):
                col_index= [self._attr2col_index(c) if isinstance(c,str) else c for c in col_index]
        
        col_index=col_index if isinstance(col_index,slice) else self._atleast_1d(col_index)
        row_index=row_index if isinstance(row_index,slice) else self._atleast_1d(row_index)
    
        index= itemgetter(*col_index)(self.tensor)
    
        if isinstance(index,tuple) is False:
            index= [index]
        return TraderDict([arr.__getitem__(row_index) for arr in index],
                         attrs=self.attrs[col_index],name=self.name)
    
    def tmatrix(self,attrs=None):
        if attrs is None:
            attrs= self.attrs
        if isinstance(attrs,str):
                col_index= self._attr2col_index(attrs)
        elif hasattr(attrs,'__len__'):
            col_index= [self._attr2col_index(c) if isinstance(c,str) else c for c in attrs]
            
        col_index=col_index if isinstance(col_index,slice) else self._atleast_1d(col_index)
        
        return itemgetter(*col_index)(self.tensor)

    def sparse_tmatrix(self,fill=0):
        I = DateIndex(self._get_dates(),self.tensor)
        I.to_matrix(fill)
        attrs= ['_'.join(col).strip() for col in I.columns]

        return TraderMatrix(data=I.matrices,
            date_info=I.date_index,attrs=attrs,name=self.name)

    def _get_dates(self):
        return [arr.date_info.dates for arr in self.tensor]
        
    def to_df(self,fill=np.nan):
        I = DateIndex(self._get_dates(),self.tensor)
        I.to_matrix(fill=fill)
        return pd.DataFrame(I.matrices, index= I.date_index,columns = pd.MultiIndex.from_tuples(I.columns))





# def from_pd(s):
#     type_v=type(s)
#     if type_v==pd.core.frame.DataFrame:
#         if s.shape[1] > 1:
#             dic= {}
#             for k, v in s.items():
#                 dic[k]= TraderArray(v.values,v.index,name=k)
#             return TraderDict(dic,name='dataframe')
#         else:
#             return TraderArray(s.values,s.index,name=s.name)
#     elif type_v == pd.core.series.Series:
#         return TraderArray(s.values,s.index,name=s.name)
#     else:
#         raise TraderError('Type {} invalid.'.format(type_v))




