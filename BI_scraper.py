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

print('loading tokenizer and Bert model')
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
BERTmodel = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
classifier = pipeline('text-classification',model=BERTmodel,tokenizer=tokenizer,truncation=True, max_length=512)

# with ['NVIDIA','META','GOOG']  and 180 pages each: it scrapes 14226 articles

# e.g. for meta are 184 pages ~ 2100days ago ~ 5years
# stock a in BI and keywords
stocks_dct = {
    'NVDA': [
        'NVIDIA', 'NVIDIA Corp', 'AI Chips', 'GPU', 'Jensen Huang', 
        'Data Center', 'Gaming', 'Blackwell', 'H200', 'B200', 'CUDA', 
        'DeepSeek', 'Groq', 'Inference', 'Export Controls', 'China Sanctions'
    ],
    'AMD': [
        'Advanced Micro Devices', 'AMD', 'Instinct', 'MI300', 'MI325X', 
        'Lisa Su', 'Ryzen', 'EPYC', 'Radeon', 'AI Accelerator', 
        'ROCm', 'OpenAI partnership', 'Market Share Gain'
    ],
    'META': [
        'Meta Platforms', 'Facebook', 'Social Media', 'Metaverse', 
        'Instagram', 'WhatsApp', 'Mark Zuckerberg', 'Llama', 'Open Source AI', 
        'Ad Revenue', 'Reality Labs', 'CapEx', 'AI Infrastructure', 'Threads'
    ],
    'GOOG': [
        'Alphabet Inc', 'Google', 'Search Engine', 'YouTube', 'Sundar Pichai', 
        'Cloud Computing', 'Waymo', 'Gemini', 'TPU', 'Vertex AI', 
        'Antitrust', 'DOJ', 'Monopoly', 'Search Generative Experience'
    ],
    'ASML': [
        'ASML', 'Lithography', 'EUV', 'High-NA', 'DUV', 'Christophe Fouquet', 
        'Semiconductor Equipment', 'TSMC', 'Intel Foundry', 'Samsung', 
        'Immersion', 'Chip Manufacturing', 'China Revenue'
    ]
}

tech_crash_events = {
    'META': [
        "2022-02-03", # -26.4% drop: First-ever user decline & Metaverse concerns
        "2022-10-27", # -24.5% drop: Massive CapEx spending/Reality Labs backlash
        "2025-10-30",  # -12% drop: AI infrastructure spending fears
        "2022-09-13", # Major CPI-day sell-off (Inflation panic)
        "2024-08-05", # Global "Carry Trade" unwind / Japanese Yen shock
        "2025-04-02"  # "Liberation Day" - massive sell-off due to tariff announcements
    ],
    'NVDA': [
        "2024-09-03", # -9.5% drop: "AI Bubble" concerns + DOJ antitrust news
        "2025-01-07", # -13% drop: Early January 2025 tech sector rotation
        "2025-01-27", # -17% drop: The "DeepSeek Shock" - largest market cap loss in history
        "2022-09-13", # Major CPI-day sell-off (Inflation panic)
        "2024-08-05", # Global "Carry Trade" unwind / Japanese Yen shock
        "2025-04-02"  # "Liberation Day" - massive sell-off due to tariff announcements
    ],
    'AMD': [
    "2022-10-06", # -14% drop: Preliminary revenue warning due to PC market slump.
    "2022-10-13", # The 2-year low ($54.57) bottom following sustained bearish sentiment.
    "2023-05-03", # -9.2% drop: Despite beating earnings, weak margins triggered a "sell the news" event.
    "2024-07-17", # -10.2% drop: Worst single-day drop in 3 years (Export curbs + sector rotation).
    "2025-11-06", # -7.3% drop: Recent "Guidance Shock" that wiped out month-to-date gains.
    "2025-11-25", # -4.2% drop: Peak of its "worst month in 3 years" (Competitive pressure from Google TPUs).
    "2022-09-13", # Major CPI-day sell-off (Inflation panic).
    "2024-08-05", # Global "Carry Trade" unwind / Japanese Yen shock.
    "2025-04-02"  # "Liberation Day" - massive sell-off due to tariff announcements.
    ],
    'GOOG': [
        "2023-02-08", # -7% drop: Bard AI demo failure
        "2023-10-25", # -9.5% drop: Cloud growth disappointment vs Azure
        "2025-02-05",  # -8% drop: DeepSeek-related efficiency concerns for AI hyperscalers
        "2022-09-13", # Major CPI-day sell-off (Inflation panic)
        "2024-08-05", # Global "Carry Trade" unwind / Japanese Yen shock
        "2025-04-02"  # "Liberation Day" - massive sell-off due to tariff announcements
    ],

    'ASML': [
    "2022-06-13", # Tech-wide sell-off + supply chain disruption fears
    "2022-10-07", # US export restrictions to China announced; ASML fell ~9%
    "2024-07-17", # -12% drop: Geopolitical tension and export curb reports
    "2024-10-15", # -16% drop: The "Accidental Leak" - Q3 earnings published early with weak guidance
    "2025-04-17", # -10% drop: Q1 2025 earnings warning about 2026 growth uncertainty
    "2025-07-16", # -8% drop: Second warning on 2026 outlook and Trump-era tariff threats
    "2022-09-13", # Major CPI-day sell-off (Inflation panic)
    "2024-08-05", # Global "Carry Trade" unwind / Japanese Yen shock
    "2025-04-02"  # "Liberation Day" - massive sell-off due to tariff announcements
],
    # 'ALL': [
    #     "2022-09-13", # Major CPI-day sell-off (Inflation panic)
    #     "2024-08-05", # Global "Carry Trade" unwind / Japanese Yen shock
    #     "2025-04-02"  # "Liberation Day" - massive sell-off due to tariff announcements
    # ]
}
def add_sub_date(date_str):
    base_date = pd.to_datetime(date_str)
    
    day_plus = (base_date + pd.Timedelta(days=1)).strftime('%m-%d-%Y')
    day_minus = (base_date - pd.Timedelta(days=1)).strftime('%m-%d-%Y')
    
    return [day_minus,base_date.strftime('%m-%d-%Y'), day_plus]


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

    return nltk.sent_tokenize(article.text)

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


for stock,keywords in stocks_dct.items():
    print(f'\nWorking on {stock}\n')

    # maybe to make it go indefinetly with a while loop, when it encounter an exception goes to next stock 
    for page in tqdm(range(80,300)): 
        
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
                art = get_article(fxd_lnk) # article split by sentences
                keysent = return_keySentences(art,keywords)
            except:
                continue

            pandas_dct['datetime'].append(dt)
            pandas_dct['label'].append(stock)
            pandas_dct['title'].append(ttl)
            pandas_dct['source'].append(src)
            pandas_dct['link'].append(fxd_lnk)

            pandas_dct['article'].append(art)
            pandas_dct['key_sentences'].append(keysent)
        
            counter += 1


## too slow, too speed up with concurrency


df = pd.DataFrame(pandas_dct).drop_duplicates(['label','link'])

df[['fb_positive','fb_neutral','fb_negative','sentiment']] = df['key_sentences'].apply(finbert_sentiment).apply(pd.Series)

df.to_csv('scraped_data.csv')
print(f"The scraper went through {counter} articles' headers")
