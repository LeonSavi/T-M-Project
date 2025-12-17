from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm # FYI this library slows processes

# with ['NVIDIA','META','GOOG']  and 180 pages each: it scrapes 14226 articles

# e.g. for meta are 184 pages ~ 2100days ago ~ 5years
stocks_lst = ['NVIDIA','META','GOOG'] 

pandas_dct = {
    'datetime':[],
    'label':[],
    'title':[],
    'source':[],
    'link':[],
    'top_sentiment':[],
    'sentiment_score':[]
    }

## to add fin bert for text analytics

counter = 0

for stock in stocks_lst:
    # maybe to make it go indefinetly with a while loop, when it encounter an exception goes to next stock 
    for page in tqdm(range(1,180+1)): 
        
        print(f'\nWorking on {stock}\n')
        url = f'https://markets.businessinsider.com/news/{stock.lower()}-stock?p={page}'

        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html,'lxml')

        articles = soup.find_all('div',class_='latest-news__story')

        for art in articles:

            dt = art.find('time', class_='latest-news__date').get('datetime')
            ttl = art.find('a',class_='news-link').text
            src = art.find('span',class_='latest-news__source').text
            lnk = art.find('a',class_='news-link').get('href')

            pandas_dct['datetime'].append(dt)
            pandas_dct['label'].append(stock)
            pandas_dct['title'].append(ttl)
            pandas_dct['source'].append(src)
            pandas_dct['link'].append(lnk)

            pandas_dct['top_sentiment'].append('')
            pandas_dct['sentiment_score'].append(0)

            counter += 1


df = pd.DataFrame(pandas_dct)
df.to_csv('scraped_data.csv')
print(f"The scraper went through {counter} articles' headers")
