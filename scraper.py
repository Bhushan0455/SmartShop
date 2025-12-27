# scraper.py - Web scraping functions
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from config import HEADERS, SCRAPING_CONFIG

def clean_price(price_text):
    """Clean and convert price text to integer"""
    if not price_text:
        return 0
        
    # Remove currency symbols and commas
    cleaned = re.sub(r'[^\d]', '', str(price_text))
    try:
        return int(cleaned)
    except:
        return 0

def extract_product_info(item, store):
    """Extract product information from HTML element"""
    try:
        if store == 'amazon':
            # Title
            title_elem = item.select_one('h2 a span')
            title = title_elem.text.strip() if title_elem else "Unknown Product"
            
            # Price
            price_elem = item.select_one('.a-price-whole')
            price = clean_price(price_elem.text) if price_elem else 0
            
            # Link
            link_elem = item.select_one('h2 a.a-link-normal')
            link = "https://www.amazon.in" + link_elem['href'] if link_elem else "#"
            
        elif store == 'flipkart':
            # Title
            title_elem = item.select_one('a[title]')
            title = title_elem['title'] if title_elem else "Unknown Product"
            
            # Price
            price_elem = item.select_one('div._30jeq3')
            price = clean_price(price_elem.text) if price_elem else 0
            
            # Link
            link_elem = item.select_one('a[href*="/p/"]')
            link = "https://www.flipkart.com" + link_elem['href'] if link_elem else "#"
        
        # Only return products with valid prices and titles
        if price > 0 and title != "Unknown Product":
            return {
                "store": store.capitalize(),
                "productName": title,
                "price": price,
                "url": link,
                "relevanceScore": 10
            }
            
    except Exception as e:
        print(f"Error extracting {store} product: {e}")
        
    return None

def scrape_amazon(search_term):
    """Scrape Amazon for products"""
    try:
        params = {'k': search_term}
        
        response = requests.get(
            'https://www.amazon.in/s',
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        product_items = soup.select('[data-component-type="s-search-result"]')
        
        products = []
        for item in product_items[:3]:  # Limit to 3 products
            product = extract_product_info(item, 'amazon')
            if product:
                products.append(product)
        
        print(f"📦 Amazon found {len(products)} products")
        return products
        
    except Exception as e:
        print(f"Amazon scraping error: {e}")
        return []

def scrape_flipkart(search_term):
    """Scrape Flipkart for products"""
    try:
        params = {'q': search_term}
        
        response = requests.get(
            'https://www.flipkart.com/search',
            params=params,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        product_items = soup.select('[data-id]')
        
        products = []
        for item in product_items[:3]:  # Limit to 3 products
            product = extract_product_info(item, 'flipkart')
            if product:
                products.append(product)
        
        print(f"📦 Flipkart found {len(products)} products")
        return products
        
    except Exception as e:
        print(f"Flipkart scraping error: {e}")
        return []