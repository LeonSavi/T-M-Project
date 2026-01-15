from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm # FYI this library slows processes
import nltk
from nltk import word_tokenize

from transformers import (AutoModelForSequenceClassification,
                          pipeline,
                          AutoTokenizer)

from newspaper import Article

from datasets import Dataset

import torch
import torch.nn.functional as F
from collections import defaultdict

from StockAnomaly import StockAnomaly

print('loading tokenizer and Bert model')
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
BERTmodel = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
classifier = pipeline('text-classification',model=BERTmodel,tokenizer=tokenizer,truncation=True, max_length=512)

# with ['NVIDIA','META','GOOG']  and 180 pages each: it scrapes 14226 articles

stocks = ["AMD", "ASML", "GOOG", "META", "NVDA"]

stock_anomaly = StockAnomaly()

stock_anomaly.calculate(window=20,price_z=2,volume_z=2)

tech_crash_events = stock_anomaly.get_anomalies(get_dict_dates=True)


def add_sub_date(date_str):
    base_date = pd.to_datetime(date_str)
    
    day_plus = (base_date + pd.Timedelta(days=7)).strftime('%m-%d-%Y')
    day_minus = (base_date - pd.Timedelta(days=7)).strftime('%m-%d-%Y')
    
    return [day_minus,base_date.strftime('%m-%d-%Y'), #day_plus
            ]


search_dates = defaultdict(list)

for k,v in tech_crash_events.items():
    temp = []
    for i in v:
        temp.extend(add_sub_date(i))
    
    search_dates[k].extend(temp)

# print(search_dates)

pandas_dct = defaultdict(list)

root_url = 'https://markets.businessinsider.com'

## to add fin bert for text analytics

counter = 0

def get_article(url:str):

    article = Article(url)
    article.download()

    article.parse()

    txt = article.text

    paragraphs = [p.strip() for p in txt.split('\n') if p.strip()]
    
    paragraph = paragraphs[0] if paragraphs else ""
    
    return txt,paragraph

def return_keySentences(sentences: list, keywords: list):
    keywords_lower = [kw.lower() for kw in keywords]
    key_sentences = []

    for sent in sentences:
        sent_lower = sent.lower()
        
        if any(kw in sent_lower for kw in keywords_lower):
            key_sentences.append(sent.strip())
    
    return ' '.join(key_sentences)


def finbert_sentiment(text:str) -> tuple[float,float,float,str]:
    '''https://www.youtube.com/watch?v=FRDKeNEeNAQ&t=640s'''
    with torch.no_grad():
 
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512     
        )
        
        inputs = {k: v.to(BERTmodel.device) for k, v in inputs.items()}

        outputs = BERTmodel(**inputs)
        
        probs = F.softmax(outputs.logits, dim=-1).squeeze()
        
        id2label = BERTmodel.config.id2label
        scores = {id2label[i]: probs[i].item() for i in range(len(id2label))}

        return (scores['positive'],scores['neutral'],scores['negative'],max(scores,key=scores.get))


for stock in stocks:
    print(f'\nWorking on {stock}\n')

    # maybe to make it go indefinetly with a while loop, when it encounter an exception goes to next stock 
    for page in tqdm(range(1,500+1)): 
        
        url = f'{root_url}/news/{stock.lower()}-stock?p={page}'

        response = requests.get(url)
        html = response.text
        soup = BeautifulSoup(html,'lxml')

        articles = soup.find_all('div',class_='latest-news__story')

        for art in articles:
            dt = art.find('time', class_='latest-news__date').get('datetime')

            check = pd.to_datetime(dt, format='%m/%d/%Y %I:%M:%S %p').strftime('%m-%d-%Y')
            if check not in search_dates[stock]:
                '''we get only the stocks we want'''
                continue

            ttl = art.find('a',class_='news-link').text
            src = art.find('span',class_='latest-news__source').text
            lnk = art.find('a',class_='news-link').get('href')

            fxd_lnk = f'{root_url}{lnk}' if lnk.startswith('/news/') else None
            
            if fxd_lnk is None:
                continue
            
            try:
                art,paragraph = get_article(fxd_lnk) # article split by sentences
                # keysent = return_keySentences(art,keywords)
            except:
                continue

            pandas_dct['datetime'].append(dt)
            pandas_dct['label'].append(stock)
            pandas_dct['title'].append(ttl)
            pandas_dct['source'].append(src)
            pandas_dct['link'].append(fxd_lnk)

            pandas_dct['article'].append(art)
            pandas_dct['fist_parag'].append(paragraph)
            # pandas_dct['key_sentences'].append(keysent)
        
            counter += 1


df = pd.DataFrame(pandas_dct).drop_duplicates(['label','link'])

# df[['fb_positive','fb_neutral','fb_negative','sentiment']] = df['fist_parag'].apply(finbert_sentiment).apply(pd.Series)

df.to_csv('scraped_data.csv')
print(f"The scraper went through {counter} articles' headers")
