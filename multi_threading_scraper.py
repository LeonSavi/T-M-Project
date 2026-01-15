import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from newspaper import Article
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configuration
STOCKS = ["AMD", "ASML", "GOOG", "META", "NVDA"]
PAGES_PER_STOCK = 500
MAX_WORKERS = 20  # Number of concurrent threads
ROOT_URL = 'https://markets.businessinsider.com'

# Thread-safe storage
results_lock = threading.Lock()
pandas_dct = defaultdict(list)

def get_article_content(url):
    """Downloads and parses a single article."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        txt = article.text
        paragraphs = [p.strip() for p in txt.split('\n') if p.strip()]
        first_paragraph = paragraphs[0] if paragraphs else ""
        return txt, first_paragraph
    except Exception:
        return None, None

def process_article_metadata(stock, art_soup):
    """Extracts metadata from the list page snippet."""
    try:
        dt = art_soup.find('time', class_='latest-news__date').get('datetime')
        ttl = art_soup.find('a', class_='news-link').text.strip()
        src = art_soup.find('span', class_='latest-news__source').text.strip()
        lnk = art_soup.find('a', class_='news-link').get('href')
        
        fxd_lnk = f'{ROOT_URL}{lnk}' if lnk.startswith('/news/') else lnk
        return {
            'datetime': dt,
            'label': stock,
            'title': ttl,
            'source': src,
            'link': fxd_lnk
        }
    except Exception:
        return None

def scrape_stock(stock):
    """Scrapes all pages for a specific stock and then fetches article contents in parallel."""
    print(f"\nGathering links for {stock}...")
    session = requests.Session()
    all_metadata = []

    # Phase 1: Collect all article links from the 500 pages
    for page in tqdm(range(1, PAGES_PER_STOCK + 1), desc=f"Pages ({stock})"):
        url = f'{ROOT_URL}/news/{stock.lower()}-stock?p={page}'
        try:
            response = session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            articles = soup.find_all('div', class_='latest-news__story')
            
            for art in articles:
                meta = process_article_metadata(stock, art)
                if meta:
                    all_metadata.append(meta)
        except Exception as e:
            continue

    print(f"Found {len(all_metadata)} articles for {stock}. Starting content download...")

    # Phase 2: Multi-threaded content download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_meta = {executor.submit(get_article_content, m['link']): m for m in all_metadata}
        
        for future in tqdm(as_completed(future_to_meta), total=len(all_metadata), desc=f"Content ({stock})"):
            meta = future_to_meta[future]
            full_text, first_para = future.result()
            
            if full_text:
                with results_lock:
                    pandas_dct['datetime'].append(meta['datetime'])
                    pandas_dct['label'].append(meta['label'])
                    pandas_dct['title'].append(meta['title'])
                    pandas_dct['source'].append(meta['source'])
                    pandas_dct['link'].append(meta['link'])
                    pandas_dct['article'].append(full_text)
                    pandas_dct['fist_parag'].append(first_para)

if __name__ == "__main__":
    # Scrape stocks sequentially but articles within them in parallel
    for stock in STOCKS:
        scrape_stock(stock)

    # Convert to DataFrame and cleanup
    df = pd.DataFrame(pandas_dct).drop_duplicates(['label', 'link'])
    
    # Save results
    # df.to_csv('scraped_data_full.csv', index=False)
    df.to_parquet("scraped_data.parquet", compression="snappy")

    print(f"\nDone! Scraped a total of {len(df)} unique articles.")

