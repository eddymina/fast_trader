# from .arr import TraderArray 
__version__ = "0.0.1"

from .stocks import YahooStock
from .webreader import YahooNews
from .utils import TraderError, TraderWarning
from .dates import DATE_FORMAT,Dates,Date,date2str,str2date,guess_datetimef,year
# from .arr import TraderArray
from .viz import trader_fig
from .trending import Trending 
from .tensor_ops import TraderArray,TraderMatrix,TraderDict

