#web reader 

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from .utils import concat_data
from .dates import str2date,date2str
import feedparser,pandas as pd 

class _News:
    def __init__(self,title=None,summary=None,time_info=None,link=None):
        self.title=title
        self.summary= summary
        self.time_info= str2date(date2str(pd.to_datetime(time_info)))
        self.link= link
        
    def to_dict(self):
        return {'title':self.title,
               'summary':self.summary,
               'time_info':self.time_info,
               'link':self.link}

class News:
    def __init__(self,news_list):
   
        d= concat_data([n.to_dict() for n in news_list])
        self.titles=d['title']
        self.summaries= d['summary']
        self.time_info= d['time_info']
        self.links= d['link']

class FinWiz:
    """
    >> FW= FinWiz(XLK)
    >> FW.parse(0).summary
    >> FW.results().summaries 
    
    """
    def __init__(self,ticker):
        finwiz_url = 'https://finviz.com/quote.ashx?t='

        url = finwiz_url + ticker
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
        response = urlopen(req)    
        # Read the contents of the file into 'html'
        self.html = BeautifulSoup(response)
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = self.html.find(id='news-table')
        # Add the table to our dictionary
        self.news_table=news_table.findAll('tr')
        self.num_res= len(self.news_table)

    def parse(self,ind):
        x=self.news_table[ind]
        text = x.a.get_text() 
        link= x.a.get('href')
        # splite text in the td tag into a list 
        date_time = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element
        if len(date_time) == 1:
            date_time=None
        else:
        	date_time=date_time[0]
        # else load 'date' as the 1st element and 'time' as the second    

        return _News(text,None,link,date_time)
    
    def results(self):
        return News([self.parse(i) for i in range(self.num_res)])

        
class YahooNews:
    
    def __init__(self,stock):
        self.stock=stock
        self.YAHOO_URL = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'
        self.feed = feedparser.parse(self.YAHOO_URL % stock)
    def parse(self):
        return News([_News(e.title,e.summary,e.published,e.link) for e in self.feed.entries])
            