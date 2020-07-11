# Welcome to `fast_trader`

#### Fast Trader is a simple yet fast deeplearning and statistics ready library for trading RND projects. The layout is simple clean and understandable. This is the first release. Signifcant model improvements will happend over time.

---

#### Getting started (Docs coming soon)

Clone this [repo](https://github.com/eddymina/fast_trader.git). This will eventually be pip installable. 

Use this quick start notebook [repo](https://github.com/eddymina/fast_trader/blob/master/Quick%20Start.ipynb). This will eventually be pip installable. 


```python

import numpy as np 
import fast_trader as ft

# interval arguments must be in '1m','2m','5m','15m','60m','90m','1d','5d','1wk','1mo','3mo'

# date indexing for all date function is quick 
### note that all dates strings are interpreted in real time. Input can be a datetime as well
# '01-10-2020'=='2020-01-10'=='01/10/2020'=='01.10.2020'

msft= ft.YahooStock('MSFT',start=None,end=None,interval='1d').history(adjusted=False)
msft
```

### Easy modular design for statistical analysis 

```python 
msft.scatter(s=10,color='violet',zorder=1) #full support for matplotlub args 

msft.close.plot()

msft.close.date_range('01-10-2010')[::1]\
		.fourier_decomp(ncomps=23)\
		.mvg_avg(window=19)\
		.greater_than(10)\
		.less_than(10000) * 2\
		.buy_hold(nshares=100,plot=True)
		# .dropna(inplace=0,newarray=True)\# will not return new array if no nans found					# and many many more features 

```
### Uses numpy backend for 5x indexing than pandas but numpy and panda friendly. 

```python
#numpy ops work here and preseve data information if they match 
close=msft.close

print(close>0)
print(close[0:100]/close[100:200])
print(2/close[100:200])
print(close[0:100]+close[100:200])
print(close[0:100]-close[100:200])


#split on date, index, or %
close.split_on('01-10-2020'),close.split_on(1000),close.split_on(.6)

#simple stats 
close.mean() #mean 
close.std() #standard deviation 
close.var() #variance 

#easily go to numpy 
close.to_numpy()
np.array(close)
close.data

#multiple ways to convert to df as well 
pd.DataFrame(close) 
close.to_df()
```