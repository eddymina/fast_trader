#helpers 
# from datetime import datetime
import importlib
import sys 
import warnings 
import requests
import bs4 as bs
import numpy as np

class TraderError(Exception):
    """
    Class specific errors 
    """
    def __dim_err__(inp):
    	if np.ndim(inp)!=1:
        	raise TraderError('Trader_array function work only on 1D Array | Try Flattening*')
    def __date_info__():
        raise TraderError('No Date information is provided. Please Add Date Info to use this functions.')
    def __date_range__(date=None,start=None,end=None):
        raise TraderError('Date information {} out of range:: {},{}'.format(date,start,end))
    def __date_type__():
        raise TraderError('Datetypes are invalid. Type Required')

class TraderWarning:
    """
    Trader Related warning 
    """
    def __init__(self,msg):
        warnings.formatwarning = self.custom_formatwarning
        self._warn(str(msg))
    def custom_formatwarning(self,msg,*args, **kwargs):
        return str(msg) + '\n'
    def _warn(self,msg):
        warnings.warn(msg)
    def __repr__(self):
        return ''

class isTrader:
    def __init__(self,*args):
        if type(args[0]) is list:
            args=args[0]
        for A in args:
            if hasattr(A, 'trader_array') is True:
                self.res = True 
            else:
                self.res = False 
                self.type= type(A)
                break 
                
    def warn(self):
        if self.res is False:
            TraderWarning('All Input(s) must be type trader.')
    def error(self):
        if self.res is False:
            raise TraderError('All Input(s) must be type trader, not type: {}'.format(self.type))

def sp500_tickers(reload=False):
    if reload:
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')
            tickers.append(ticker[0].text.split('\n')[0])

        return np.array(tickers)
    else:
        return np.array(['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP',
       'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK', 'ALB', 'ARE', 'ALXN',
       'ALGN', 'ALLE', 'ADS', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN',
       'AMCR', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP',
       'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'AOS',
       'APA', 'AIV', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ',
       'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'BKR', 'BLL',
       'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIIB', 'BLK', 'BA',
       'BKNG', 'BWA', 'BXP', 'BSX', 'BMY', 'AVGO', 'BR', 'BF.B', 'CHRW',
       'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT',
       'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CTL', 'CERN', 'CF',
       'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS',
       'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH',
       'CL', 'CMCSA', 'CMA', 'CAG', 'CXO', 'COP', 'ED', 'STZ', 'COO',
       'CPRT', 'GLW', 'CTVA', 'COST', 'COTY', 'CCI', 'CSX', 'CMI', 'CVS',
       'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM',
       'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH', 'DG', 'DLTR', 'D',
       'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'ETFC',
       'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMR', 'ETR',
       'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'EVRG', 'ES', 'RE',
       'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT',
       'FDX', 'FIS', 'FITB', 'FE', 'FRC', 'FISV', 'FLT', 'FLIR', 'FLS',
       'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX',
       'GPS', 'GRMN', 'IT', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL',
       'GPN', 'GS', 'GWW', 'HRB', 'HAL', 'HBI', 'HOG', 'HIG', 'HAS',
       'HCA', 'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX',
       'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'HII',
       'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE',
       'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV',
       'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR',
       'KSU', 'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS',
       'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG', 'LDOS',
       'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LYB',
       'MTB', 'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA',
       'MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM',
       'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MNST', 'MCO',
       'MS', 'MOS', 'MSI', 'MSCI', 'MYL', 'NDAQ', 'NOV', 'NTAP', 'NFLX',
       'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NBL',
       'JWN', 'NSC', 'NTRS', 'NOC', 'NLOK', 'NCLH', 'NRG', 'NUE', 'NVDA',
       'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS', 'PCAR',
       'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI',
       'PRGO', 'PFE', 'PM', 'PSX', 'PNW', 'PXD', 'PNC', 'PPG', 'PPL',
       'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH',
       'QRVO', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG',
       'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL', 'ROP', 'ROST',
       'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW',
       'SHW', 'SPG', 'SWKS', 'SLG', 'SNA', 'SO', 'LUV', 'SWK', 'SBUX',
       'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW',
       'TTWO', 'TPR', 'TGT', 'TEL', 'FTI', 'TFX', 'TXN', 'TXT', 'TMO',
       'TIF', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TFC', 'TWTR', 'TSN',
       'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS',
       'URI', 'UHS', 'UNM', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK',
       'VZ', 'VRTX', 'VIAC', 'V', 'VNO', 'VMC', 'WRB', 'WAB', 'WMT',
       'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC',
       'WU', 'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XRX',
       'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS'], dtype='<U5')


def concat_data(dicts):
	"""	
	Concat a list of dicts 
	"""
	d = {}
	for k in dicts[0].keys():
	    d[k] = tuple(d[k] for d in dicts)
	return d 

def reload(pkg):
	importlib.reload(pkg)

def progress_bar(value, endvalue, bar_length=35):
    """
    Simple progress bar 

    value:: current value in loop
    endvalue:: final value (-1 for indexing)
    bar_length:: # of - to use 

    returns progress 

    for i in range(10):
        progress_bar(i,10)
    """
    endvalue-=1
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rProgress: [{0}] {1}% | {2}/{3}".format(arrow + spaces, int(round(percent * 100)),
                                                          value+1,endvalue+1))
    sys.stdout.flush()
    
    if value == endvalue:
        print(' Done.')

def time_line(code_str,print_line=True,cuml_time=False,precision=8):
    '''
    Computes the length a line of 
    code. Only support for simple code-
    doesn't work for tabs + spaces.

    code_str:: string of code newline split 
    print_line:: will print line with run time
    cuml_time:: boolean (show cumlative time)
    precision:: number of decimal pts 
    
    ie change if True:  ---> if True: b=a//2
                  b=a//2
    example::
    
    ## running from notebook (use libs not imported in string)

    >> import numpy as np 
    >> code_str="""
                np.random.randint(0,10,(1000000,1))
                10+10
                """
    >> exec(time_line(code_str,debug=True))
    
    Line 1: np.random.randint(0,10,(1000000,1)) | 0.0185 runtime
    Line 2: 10+10 | 0.0 runtime
    TTL runtime:: 0.019s
    '''
    import time
    code_str= str(code_str)
    code_str_with_time= []
    code_str_split=code_str.split('\n')
    code_str_with_time.append('import time')
    code_str_with_time.append('ttl_t=0')
    
    P=precision
    for i,s in enumerate(code_str_split):
        if s!='' and list(s)[0]!='#':
            code_str_with_time.append('s= time.time()')
            code_str_with_time.append(s)
            code_str_with_time.append('e=time.time()-s')
            code_str_with_time.append('ttl_t+=e')
            if print_line:
                if cuml_time:
                    #code_str_with_time.append('print("Line {}: {} |", round(e,{}),"runtime | cumulative_time",round(ttl_t,{}))'.format(i,s,P,P))
                    code_str_with_time.append('print("Line {}:", round(e,{}),"/",round(ttl_t,{}),"(s) | {}")'.format(i,P,P,s))
                else:
                    code_str_with_time.append('print("Line {}:", round(e,{}),"runtime (s)| {}")'.format(i,P,s))   
            else:
                if cuml_time:
                    #code_str_with_time.append('print("Line {}:", round(e,{}),"runtime | cumulative_time",round(ttl_t,{}))'.format(i,P,P))
                    code_str_with_time.append('print("Line {}:", round(e,{}),"/",round(ttl_t,{}),"(s)")'.format(i,P,P))
                else:
                    code_str_with_time.append('print("Line {}:", round(e,{}),"runtime (s)")'.format(i,P))
    code_str_with_time.append('print("TTL runtime:", round(ttl_t,{}),"(s)")'.format(P))
    return '\n'.join(code_str_with_time)

