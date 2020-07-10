import numpy as np 
from .utils import TraderError, TraderWarning, isTrader 
from .array_ops import TraderDict, TraderArray
import matplotlib.pyplot as plt 
import datetime as dt 

def fmt_dates():
    plt.gcf().autofmt_xdate()
    plt.show()

def trader_fig(figsize=(20,7)):              
    plt.figure(figsize=figsize)

class DateGroup:
    
    def __init__(self,tarr):
        isTrader(tarr).error()
        self.tarr=tarr
        
    def var_month(self,year,day=15):
        """
        For a given year 
        see how a particular month 
        performs on a certain day 
        """
        arrays={}
        dates=[]
        
        if year == self.tarr.date_info.min.year:
            TraderWarning('Avoid using years that are cut off by start or end date')
            for i in range(self.tarr.date_info.min.month+1,13):
                dates.append(dt.datetime(year=year,month=i,day=day))
                
        elif year == self.tarr.date_info.max.year:
            TraderWarning('Avoid using years that are cut off by start or end date')
            
            for i in range(1,self.tarr.date_info.max.month+1):
                dates.append(dt.datetime(year=year,month=i,day=day))
            
        elif year > self.tarr.date_info.max.year or year < self.tarr.date_info.min.year:
            raise TraderError('Year out of range.')
        else: 
            for i in range(1,13):
                dates.append(dt.datetime(year=year,month=i,day=day))
        for i in range(len(dates)-1):
            arrays[dates[i]]= self.tarr.date_range(dates[i],dates[i+1])
        return TraderDict(arrays,name='vmonthly_on_year_{}_day_{}'.format(year,day)) 
    
    def var_annual(self,month=1,day=15):
        """
        For a every year show var for 
        a particular month on a fixed day 
        
        month:: int [1,12]
        day:: int[1,31] (include all months)
        """
        arrays= {}
        dates=[]
        for i in range(self.tarr.date_info.min.year+1,self.tarr.date_info.max.year):
            dates.append(dt.datetime(year=i,month=month,day=day))
            
        dates= [self.tarr.date_info.min]+ dates+ [self.tarr.date_info.max]
        
        for i in range(len(dates)-1):
            arrays[dates[i]]= self.tarr.date_range(dates[i],dates[i+1])
        return TraderDict(arrays,name='vannual_on_month_{}_day_{}'.format(month,day))

    def plotter(self,tensor,err_color='lightgray',alpha=.2,fill_color='pink'):
        plot_data=[]
        plt.title(tensor.name)
        for arr in tensor:
            x,y= arr
            mean=y.mean()
            plt.errorbar(x, mean, fmt='--o',elinewidth=5,
                         yerr=[np.expand_dims(mean-y.min,0),
                               np.expand_dims(y.max-mean,0)], ecolor=err_color)
            plot_data.append((x,mean,y.min,y.max))
        plot_data=np.array(plot_data)
        plt.fill_between(plot_data[:,0],
                         y1=plot_data[:,2].astype(float),
                         y2=plot_data[:,3].astype(float),
                        alpha=alpha,zorder=2,color=fill_color)


def plotly_draw(tarrays):
    """
    Create Plotly object of a series of plots
    
    must have plotly installed
    
    >> plotly([tarray,tarray.mvg_avg(7),
        tarray.mvg_avg(21),tarray.fourier_decomp(35)]) 
    """
    import plotly.graph_objects as go
    isTrader(tarrays).error()
    fig = go.Figure()
    for tarray in tarrays:
        if tarray.date_info is not None:
            fig.add_trace(go.Scatter(x=tarray.date_info.dates, y=tarray.data,name=tarray.name))
        else:
            fig.add_trace(go.Scatter(x=np.arange(tarray.size), y=tarray.data,name=tarray.name))
        
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()