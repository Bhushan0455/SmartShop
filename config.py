# config.py
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# Web scraping configuration
SCRAPING_CONFIG = {
    'amazon': {
        'base_url': 'https://www.amazon.in/s',
        'params': {'k': ''},
        'selectors': {
            'products': '[data-component-type="s-search-result"]',
            'title': 'h2 a span',
            'price': '.a-price-whole',
            'link': 'h2 a.a-link-normal',
            'image': '.s-image'
        }
    },
    'flipkart': {
        'base_url': 'https://www.flipkart.com/search',
        'params': {'q': ''},
        'selectors': {
            'products': '[data-id]',
            'title': 'a[title]',
            'price': 'div._30jeq3',
            'link': 'a[href*="/p/"]',
            'image': 'img[src*="flipkart"]'
        }
    }
}

# ML Model configuration
ML_CONFIG = {
    'image_size': (224, 224),
    'brands': ['samsung', 'apple', 'sony', 'oneplus', 'xiaomi', 'google', 'realme', 'oppo', 'vivo', 'nothing', 'motorola']
}