# import numpy as np 
# import matplotlib.pyplot as plt 
# import pandas as pd 
# from math import sqrt
# from pandas.plotting import register_matplotlib_converters
# register_matplotlib_converters()

# from ..utils import TraderError,TraderWarning,isTrader
# from ..dates import DATE_FORMAT,Dates,Date,date2str,str2date,guess_datetimef

# class TraderDict(dict):
#     def __init__(self, dic,name=None):

#         core_dict= dic.copy()
#         if type(core_dict) is not dict:
#             raise TraderError('Set Input As a Dictionary')

#         self.__dict__ = core_dict
#         self.tensor_dict=True

#         self._core_dict= self._convert_keys_to_string(dic.copy())

#         self.trader_dict = True
#         self.nitems=self.ndim=len(self._core_dict)
#         self.tensor=list(self._core_dict.values())

#         self._date = True
#         for arr in self.tensor: 
#             if arr.date_info is None:
#                 self._date = False
#                 break 

#         self.shape= (self.nitems,[arr.shape[0] for arr in self.tensor])
#         self.name=name
#         self.attrs= list(self._core_dict.keys())

#     def append(self,newd):

#         self.__dict__.update(newd)

#         d2str=self._convert_keys_to_string(newd)

#         self._core_dict.update(d2str)
#         self.nitems+=len(newd)
#         self.ndim+=len(newd)

#         ntensor=list(d2str.values())

#         isTrader(ntensor).error()

#         self.tensor.extend(ntensor)

#         self._date = True
#         for arr in ntensor: 
#             if arr.date_info is None:
#                 self._date = False
#                 break 

#         self.shape[1].extend([arr.shape[0] for arr in ntensor])
#         self.attrs.extend(list(d2str.keys()))

#     def __repr__(self):
#         """
#         Representation of array. 
#         """
#         base= '<TraderDict:: shape=({},{}),'.format(self.shape[0],self.shape[1])
#         if self.nitems>5:
#             base+=' attrs={}...,'.format(self.attrs[:4])
#         else:
#             base+=' attrs={},'.format(self.attrs)

#         base+=' name={}'.format(self.name)

#         return base+'>'
    
#     def __len__(self):
#         return self.nitems
    
#     def __iter__(self):
#         """
#         Trader array is iterable
#         """
#         for i in range(self.nitems):
#             yield(self.attrs[i],self.tensor[i])

#     def index_attrs(self,i):
#         type_i =type(i)
#         if np.ndim(i) > 1:
#             raise TraderError('Only 1D indexing supported. Use .to_numpy() or to_df() for index broadcasting.')
#         if hasattr(i,'__len__'):
#             if type(i[0]) is str:
#                 return TraderDict(dict([(a,self.__dict__[a]) for a in i]),
#                                   name=self.name)
#             return TraderDict(dict([(self.attrs[a],self.tensor[a]) for a in i])
#                                        ,name=self.name)
#         elif type_i is str:
#             return self.__dict__[i]
#         elif type_i is slice:
#             return TraderDict(dict(zip(self.attrs[i],self.tensor[i])))
#         elif type_i is int:
#             return TraderDict(dict(zip([self.attrs[i]],[self.tensor[i]])),name=self.name)
#         else:
#             raise TraderError('Invalid indexing operation.')

#     def _get_x(self,i):
#         f='__getitem__'
#         return TraderDict(self._arrayf(f,i=i),name=self.name)

#     def __getitem__(self,i):
#         f='__getitem__'

#         if isinstance(i, tuple):
#             assert len(i)==2,'only support for 2D multiindex'
#             i1,i2= i 

#             if i1 and i2 is not None:
#                 if isinstance(i1, slice) and type(i1.start) != int:
#                     s= self.date_range(start=i1.start,end=i1.stop)
#                 else:
#                     s= self._get_x(i1)

#                 return s.index_attrs(i2)

#             if i1 is not None:
#                 if isinstance(i1, slice) and type(i1.start) != int:
#                     return self.date_range(start=i1.start,end=i1.stop)
#                 return self._get_x(i1)

#             if i2 is not None:
#                 return self.index_attrs(i2)

#             if type(i.start) != int:
#                 return self.date_range(start=i.start,end=i.stop)
#             return self._get_x(i1)

#         if isinstance(i, list) or isinstance(i, int):
#             return self._get_x(i)
#         raise TraderError('Invalid Index(s)')

#     def to_dict(self):
#         return self._core_dict
            
#     def to_numpy(self):

#         def fill_nan(arr,n,nan_type):
#             return np.concatenate([arr,np.repeat(nan_type,n-arr.shape[0])])
    
#         date_info=[]
#         arrays=[]
#         max_size= max(self.shape[1])
#         for arr in self.tensor:
#             if self._date:
#                 date_info.append(fill_nan(np.array(list(arr.date_info.dates),dtype=np.datetime64),
#                                       max_size,nan_type=np.datetime64("NaT")))

#             arrays.append(fill_nan(arr.data,max_size,nan_type=np.nan))

#         if self._date: 
#             dates= np.array(date_info,dtype=np.datetime64)
#             uniq=np.unique(dates)

#             if uniq.shape[0]==dates.shape[1]:
#                 dates=uniq
                
#             return {'date_tensor':dates,
#                         'data_tensor':np.array(arrays,dtype=np.float64)}

#         return {'data_tensor':np.array(arrays,dtype=np.float64)}
    
#     def _convert_keys_to_string(self,dictionary):
#         return dict((str(k).lower(), v) \
#                 for k, v in dictionary.items())
        
#     def to_df(self):
#         np_data =self.to_numpy()
#         index= None
#         if self._date and np_data['date_tensor'].ndim==1:
#             index= np_data['date_tensor']

#         dict_input = dict(zip(self.attrs,np_data['data_tensor']))
#         return pd.DataFrame(dict_input,index=index)
    
#     def _arrayf(self,f,*args,**kwargs):
#         output_dict={}
#         for i,name in enumerate(self.attrs):
#             e='output_dict[name] = self.tensor[i]'+'.'+f+'(*args,**kwargs)'
#             exec(e)
#         return output_dict
        
#     def fourier_decomp(self,*args,**kwargs):
#         f='fourier_decomp'
#         return TraderDict(self._arrayf(f,**kwargs),name=f)
    
#     def mvg_avg(self,*args,**kwargs):
#         f='mvg_avg'
#         return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
    
#     def num_grad(self,*args,**kwargs):
#         f='num_grad'
#         return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
        
#     def np_func(self,*args,**kwargs):
#         f='np_func'
#         return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
    
#     def minmax_scale(self,*args,**kwargs):
#         f='minmax_scale'
#         return TraderDict(self._arrayf(f,*args,**kwargs),name=f)
        
#     def standard_scale(self):
#         f='standard_scale'
#         return TraderDict(self._arrayf(f),name=f)
        
#     def buy_hold(self,*args,**kwargs):
#         f='buy_hold'
#         return self._arrayf(f,*args,**kwargs)
        
#     def dropna(self,inplace=None):
#         output_dict={}
#         for i in range(self.ndim):
#             self.tensor[i]
#             out = self.tensor[i].dropna(inplace=inplace,newarray=True)
#             if out is None:
#                 output_dict[self.attrs[i]] = self.tensor[i]
#             else:
#                 output_dict[self.attrs[i]] = out 
#         self.__init__(output_dict,name=self.name)
        
#     def plot(self,*args,**kwargs):
#         f='plot'
#         self._arrayf(f,*args,**kwargs)
        
#     def scatter(self,*args,**kwargs):
#         f='scatter'
#         self._arrayf(f,*args,**kwargs)
    
#     def date_range(self,*args,**kwargs):
#         f='date_range'
#         return TraderDict(self._arrayf(f,*args,**kwargs),name=f)

#     def index_dates(self,*args,**kwargs):
#         f='index_dates'
#         return TraderDict(self._arrayf(f,*args,**kwargs),name=f)