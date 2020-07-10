import requests
import bs4 as bs
import pandas as pd
import numpy as np 

class Trending:
    def __init__(self):
        self.industries=['basic_materials','communication_services',
                            'consumer_defensive','consumer_cyclical',
                           'energy','healthcare','financial_services','industrials',
                           'real_estate','technology']
    
    def ticks(self):
        return self._base('https://finance.yahoo.com/trending-tickers',4,5,12)
       
    def _base(self,url,ind1=3,ind2=4,mod=10):
        resp = requests.get(url)
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.findAll('td')
        trending= []
        inds = [(i,i+ind1,i+ind2) for i in range(len(table)) if i%mod==0]
        for ind in inds:
            a= table[ind[0]].find('a')
            symbol= a['href'].split('=')[1]
            title= a['title']
            dollar_change= table[ind[1]].text
            percent_change= table[ind[2]].text
            trending.append((symbol,title,np.float32(dollar_change),
                             np.char.strip(percent_change,'%').astype(np.float32)))
            
        return pd.DataFrame(trending,columns=['symbol','title','currency_change','percent_change']) 

    def gainers(self):
        return self._base('https://finance.yahoo.com/gainers')
        
    def losers(self):
        return self._base('https://finance.yahoo.com/losers')

    def most_active(self):
        return self._base('https://finance.yahoo.com/most-active')
    
    def etfs(self):
        return self._base('https://finance.yahoo.com/etfs',mod=9)
    
    def mutual_funds(self):
        
        resp = requests.get('https://finance.yahoo.com/mutualfunds')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.findAll('td')
        trending= []
        inds = [(i,i+2,i+3,i+7) for i in range(len(table)) if i%10==0]
        for ind in inds:
            a= table[ind[0]].find('a')
            symbol= a['href'].split('=')[1]
            title= a['title']
            dollar_change= table[ind[1]].text
            percent_change= table[ind[2]].text
            return_3m= table[ind[2]].text
            trending.append((symbol,title,np.float32(dollar_change),
                             np.char.strip(percent_change,'%').astype(np.float32),return_3m))
            
        return pd.DataFrame(trending,columns=['symbol','title',
                                              'currency_change','percent_change','return_3m'])
    
    def industry(self,industry='technology'):
        if industry not in self.industries:
            raise TraderError('industry arg must be on of {}'.format(self.industries))
        return self._base('https://finance.yahoo.com/sector/ms_{}'.format(industry))

