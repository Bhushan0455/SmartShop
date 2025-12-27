from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import re
import os
import sqlite3
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from enhanced_brand_detector import brand_detector

# Import TensorFlow for ML
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow not available: {e}")

# Import ML modules
try:
    from train_model import ImprovedBrandClassifier, detect_brand_ml
    ML_AVAILABLE = True
    print("✅ ML modules imported successfully")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML modules not available: {e}")

# Import other modules
from Products_data import SAMPLE_PRODUCTS, BRAND_SUGGESTIONS, KNOWN_BRANDS
from config import HEADERS
from scraper import scrape_amazon, scrape_flipkart

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ML model
ML_MODEL_LOADED = False
if ML_AVAILABLE:
    try:
        # Check if model exists, if not create a simple one
        if not os.path.exists('improved_brand_model.h5'):
            print("🔄 No pre-trained model found. Training a simple model...")
            train_simple_model()
        
        ML_MODEL_LOADED = True
        print("✅ ML model initialized successfully")
    except Exception as e:
        print(f"⚠️ ML model initialization failed: {e}")
        ML_MODEL_LOADED = False

# Database setup
def init_database():
    """Initialize SQLite database for caching scraped results"""
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store TEXT NOT NULL,
            product_name TEXT NOT NULL,
            price INTEGER NOT NULL,
            url TEXT NOT NULL,
            category TEXT NOT NULL,
            search_keywords TEXT NOT NULL,
            model_number TEXT,
            exact_model_match BOOLEAN DEFAULT FALSE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute("SELECT COUNT(*) FROM products")
    count = cursor.fetchone()[0]
    
    if count == 0:
        cursor.executemany('''
            INSERT INTO products (store, product_name, price, url, category, search_keywords, model_number, exact_model_match)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', SAMPLE_PRODUCTS)
        print(f"📊 Database initialized with {len(SAMPLE_PRODUCTS)} sample products")
    
    conn.commit()
    conn.close()

init_database()

def train_simple_model():
    """Train a simple model if no pre-trained model exists"""
    try:
        from train_model import train_model_now
        print("🎯 Training brand detection model...")
        success = train_model_now()
        if success:
            print("✅ Model training completed successfully")
        else:
            print("⚠️ Model training failed, using fallback methods")
    except Exception as e:
        print(f"⚠️ Could not train model: {e}")

# Utility functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_brand_from_search(search_term):
    """Detect brand from search term using keyword matching"""
    search_lower = search_term.lower()
    for brand, keywords in KNOWN_BRANDS.items():
        if any(keyword in search_lower for keyword in keywords):
            return brand.capitalize()
    return None

def detect_brand_from_filename(filename):
    """Detect brand from filename using keyword matching"""
    filename_lower = filename.lower()
    for brand, keywords in KNOWN_BRANDS.items():
        if any(keyword in filename_lower for keyword in keywords):
            return brand.capitalize()
    return None

def detect_brand_from_image_visual(image_path):
    """Simple visual brand detection based on color patterns"""
    try:
        img = Image.open(image_path)
        img = img.resize((100, 100))
        img_array = np.array(img)
        
        # Calculate average color
        avg_color = np.mean(img_array, axis=(0, 1))
        
        if avg_color[0] > 150 and avg_color[1] < 100 and avg_color[2] < 100:  # Red dominant
            return "Oneplus"
        elif avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] > 150:  # Blue dominant
            return "Samsung"
        elif np.all(avg_color < 100):  # Dark colors
            return "Apple"
        elif avg_color[0] > 200 and avg_color[1] > 200:  # Light colors
            return "Sony"
            
    except Exception as e:
        print(f"Visual detection error: {e}")
    
    return None

def get_brand_suggestions(brand):
    """Get model suggestions for a detected brand with case-insensitive matching"""
    if not brand:
        return []
    
    # Case-insensitive matching
    brand_lower = brand.lower()
    for brand_key, suggestions in BRAND_SUGGESTIONS.items():
        if brand_key.lower() == brand_lower:
            print(f"✅ Found suggestions for {brand}: {suggestions}")
            return suggestions
    
    print(f"⚠️ No suggestions found for brand: {brand}")
    return []

def search_products_real_time(product_name):
    """Perform real-time web scraping for products with better filtering"""
    try:
        print(f"🔍 Real-time search for: '{product_name}'")
        
        # Search both Amazon and Flipkart
        amazon_results = scrape_amazon(product_name)
        flipkart_results = scrape_flipkart(product_name)
        
        all_results = amazon_results + flipkart_results
        
        # If no real-time results, use database with EXACT matching
        if not all_results:
            print("🔄 No real-time results, using enhanced database search")
            return get_products_from_database_enhanced(product_name)
        
        print(f"✅ Real-time found {len(all_results)} products")
        return all_results[:8]
        
    except Exception as e:
        print(f"Real-time search error: {e}")
        return get_products_from_database_enhanced(product_name)

def get_products_from_database_enhanced(search_term):
    """Enhanced database search that prioritizes search relevance over price"""
    try:
        conn = sqlite3.connect('products.db')
        cursor = conn.cursor()
        
        search_lower = search_term.lower().strip()
        print(f"🔍 Enhanced database search for: '{search_term}'")
        
        products = []
        seen_names = set()
        
        # Calculate relevance scores based on how well the product matches the search
        def calculate_relevance_score(product_name, search_lower):
            product_lower = product_name.lower()
            score = 0
            
            # Exact match bonus (highest priority)
            if product_lower == search_lower:
                score += 1000
            
            # Contains all search words in order
            search_words = search_lower.split()
            if all(word in product_lower for word in search_words):
                score += 500
                
                # Bonus for consecutive words in correct order
                if ' '.join(search_words) in product_lower:
                    score += 300
            
            # Brand-specific relevance for generic searches
            if search_lower == 'apple':
                if 'iphone' in product_lower:
                    score += 400  
                elif 'ipad' in product_lower:
                    score += 300
                elif 'macbook' in product_lower:
                    score += 200
                elif 'watch' in product_lower:
                    score += 100
                elif 'airpods' in product_lower:
                    score += 50
                    
            elif search_lower == 'samsung':
                if 'galaxy' in product_lower and any(model in product_lower for model in ['s25', 's24', 's23', 'fold', 'flip']):
                    score += 400  
                elif 'galaxy' in product_lower and 'a' in product_lower:
                    score += 300  
                elif 'galaxy' in product_lower and 'm' in product_lower:
                    score += 200  
                elif 'buds' in product_lower:
                    score += 100
                elif 'watch' in product_lower:
                    score += 100
                    
            elif search_lower == 'oneplus':
                if 'oneplus' in product_lower and any(model in product_lower for model in ['12', '11', '10']):
                    score += 400  
                elif 'nord' in product_lower:
                    score += 300  
                elif 'buds' in product_lower:
                    score += 100
                    
            # Specific model matching
            if any(word in search_lower for word in ['iphone', 'samsung', 'oneplus']):
                # Extract numbers from search for model matching
                import re
                search_numbers = re.findall(r'\d+', search_lower)
                product_numbers = re.findall(r'\d+', product_lower)
                
                if search_numbers and product_numbers:
                    # Bonus for matching model numbers
                    if any(num in product_numbers for num in search_numbers):
                        score += 200
            
            # Category relevance
            if 'iphone' in search_lower and 'iphone' in product_lower:
                score += 300
            if 'ipad' in search_lower and 'ipad' in product_lower:
                score += 300
            if 'watch' in search_lower and 'watch' in product_lower:
                score += 300
            if 'airpods' in search_lower and 'airpods' in product_lower:
                score += 300
                
            # Word position bonus - words at beginning get higher score
            for i, word in enumerate(search_words):
                if product_lower.startswith(word):
                    score += (len(search_words) - i) * 10
            
            return score
        
        # Search with multiple strategies
        search_pattern = f'%{search_lower}%'
        queries = [
            "SELECT store, product_name, price, url FROM products WHERE LOWER(product_name) LIKE ?",
            "SELECT store, product_name, price, url FROM products WHERE LOWER(search_keywords) LIKE ?", 
            "SELECT store, product_name, price, url FROM products WHERE LOWER(model_number) LIKE ?"
        ]
        
        # Execute all search strategies
        for query in queries:
            try:
                cursor.execute(query, (search_pattern,))
                results = cursor.fetchall()
                
                for row in results:
                    product_name = row[1]
                    
                    if product_name not in seen_names:
                        seen_names.add(product_name)
                        
                        # Calculate dynamic relevance score
                        relevance_score = calculate_relevance_score(product_name, search_lower)
                        
                        product = {
                            "store": row[0],
                            "productName": product_name,
                            "price": row[2],
                            "url": row[3],
                            "relevanceScore": relevance_score
                        }
                        products.append(product)
                        print(f"    ✅ Added: {product_name} (Score: {relevance_score})")
                
            except Exception as query_error:
                print(f"  Query error: {query_error}")
                continue
        
        conn.close()
        
        if products:
            print(f"📊 Found {len(products)} products for '{search_term}'")
            
            # SORT BY RELEVANCE SCORE ONLY - IGNORE PRICE COMPLETELY
            products.sort(key=lambda x: -x['relevanceScore'])
            
            # Debug: Show top products with scores
            print("🏆 TOP PRODUCTS SORTED BY RELEVANCE:")
            for i, p in enumerate(products[:10]):
                print(f"   {i+1}. {p['productName']} - Relevance: {p['relevanceScore']}, Price: ₹{p['price']}")
            
            return products[:6]
        else:
            print(f"📊 No products found, using fallback")
            return get_smart_fallback(search_term)
            
    except Exception as e:
        print(f"Enhanced database search error: {e}")
        return get_smart_fallback(search_term)
        
def is_relevant_product(product_name, search_term):
    """Check if a product is actually relevant to the search"""
    product_lower = product_name.lower()
    search_lower = search_term.lower()
    
    if search_lower in ['apple', 'samsung', 'oneplus']:
        if 'iphone' in product_lower or 'galaxy' in product_lower or 'oneplus' in product_lower:
            return True
        if any(accessory in product_lower for accessory in ['airpods', 'watch', 'buds', 'earphones']):
            return False
    
    if 'iphone' in search_lower and 'iphone' not in product_lower:
        return False
    
    if any(keyword in search_lower for keyword in ['samsung', 'galaxy']) and 'mobile' not in search_lower:
        if any(wrong_product in product_lower for wrong_product in ['buds', 'watch', 'tab', 'tablet']):
            return False
    
    return True

def get_smart_fallback(search_term):
    """Smart fallback that understands product categories"""
    search_lower = search_term.lower()
    
    # Apple products
    if 'iphone 16' in search_lower:
        return [
            {"store": "Amazon", "productName": "Apple iPhone 16 Pro (256GB, Space Black)", "price": 134999, "url": "https://www.amazon.in/s?k=iphone+16+pro", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Apple iPhone 16 Pro (512GB, Titanium Blue)", "price": 149999, "url": "https://www.flipkart.com/search?q=iphone+16+pro", "relevanceScore": 10},
            {"store": "Amazon", "productName": "Apple iPhone 16 (128GB, Pink)", "price": 79999, "url": "https://www.amazon.in/s?k=iphone+16", "relevanceScore": 10}
        ]
    elif 'iphone' in search_lower:
        return [
            {"store": "Amazon", "productName": "Apple iPhone 16 Pro (256GB, Space Black)", "price": 134999, "url": "https://www.amazon.in/s?k=iphone+16+pro", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Apple iPhone 16 (128GB, Pink)", "price": 79999, "url": "https://www.flipkart.com/search?q=iphone+16", "relevanceScore": 9},
            {"store": "Amazon", "productName": "Apple iPhone 15 Pro (256GB, Blue Titanium)", "price": 119999, "url": "https://www.amazon.in/s?k=iphone+15+pro", "relevanceScore": 8}
        ]
    
    # Samsung products
    elif 'samsung s24' in search_lower or 'galaxy s24' in search_lower:
        return [
            {"store": "Amazon", "productName": "Samsung Galaxy S24 Ultra (256GB, Titanium Violet)", "price": 124999, "url": "https://www.amazon.in/s?k=samsung+s24+ultra", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Samsung Galaxy S24+ (256GB, Marble Gray)", "price": 89999, "url": "https://www.flipkart.com/search?q=samsung+s24+plus", "relevanceScore": 10},
            {"store": "Amazon", "productName": "Samsung Galaxy S24 (128GB, Marble Gray)", "price": 79999, "url": "https://www.amazon.in/s?k=samsung+s24", "relevanceScore": 10}
        ]
    elif 'samsung' in search_lower or 'galaxy' in search_lower:
        return [
            {"store": "Amazon", "productName": "Samsung Galaxy S25 Ultra (512GB, Phantom Black)", "price": 139999, "url": "https://www.amazon.in/s?k=samsung+s25+ultra", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Samsung Galaxy A55 5G (128GB, Awesome Iceblue)", "price": 34999, "url": "https://www.flipkart.com/search?q=samsung+a55", "relevanceScore": 9},
            {"store": "Amazon", "productName": "Samsung Galaxy Z Fold5 (512GB, Phantom Black)", "price": 154999, "url": "https://www.amazon.in/s?k=samsung+z+fold5", "relevanceScore": 8}
        ]
    
    # OnePlus products
    elif 'oneplus' in search_lower:
        return [
            {"store": "Amazon", "productName": "OnePlus 12 (256GB, Silky Black)", "price": 69999, "url": "https://www.amazon.in/s?k=oneplus+12", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "OnePlus 12R (128GB, Iron Gray)", "price": 39999, "url": "https://www.flipkart.com/search?q=oneplus+12r", "relevanceScore": 9},
            {"store": "Amazon", "productName": "OnePlus Nord CE 3 (128GB, Aqua Surge)", "price": 24999, "url": "https://www.amazon.in/s?k=oneplus+nord+ce+3", "relevanceScore": 8}
        ]
    
    else:
        return get_fallback_products(search_term)

def search_products_by_model(brand, model):
    """Enhanced product search with better model matching"""
    try:
        # First try real-time scraping with improved query
        search_query = f"{brand} {model}"
        
        print(f"🎯 ENHANCED MODEL SEARCH: Brand='{brand}', Model='{model}', Query='{search_query}'")
        
        # Search both Amazon and Flipkart
        amazon_results = scrape_amazon(search_query)
        flipkart_results = scrape_flipkart(search_query)
        
        all_results = amazon_results + flipkart_results
        
        print(f"🌐 Real-time scraping found: {len(amazon_results)} Amazon + {len(flipkart_results)} Flipkart = {len(all_results)} total")
        
        # If no real-time results, try database with enhanced matching
        if not all_results:
            print("🔄 No real-time results, falling back to database search")
            all_results = enhanced_database_search(brand, model)
        else:
            print("✅ Using real-time scraping results")
        
        return all_results[:8]
        
    except Exception as e:
        print(f"💥 Model search error: {e}")
        return enhanced_database_search(brand, model)
        
def enhanced_database_search(brand, model):
    """Enhanced database search with better model matching"""
    try:
        conn = sqlite3.connect('products.db')
        cursor = conn.cursor()
        
        # Improved model cleaning that preserves fold/flip models
        clean_model = model.lower().strip()
        clean_brand = brand.lower().strip()
        
        print(f"🔍 Database search: brand='{brand}', model='{model}', clean_model='{clean_model}'")
        
        # Multiple search strategies with improved matching
        queries = [
            # Exact model number match (highest priority)
            f"SELECT store, product_name, price, url, '10' as relevanceScore FROM products WHERE LOWER(model_number) LIKE '%{clean_model}%'",
            # Product name contains exact model
            f"SELECT store, product_name, price, url, '9' as relevanceScore FROM products WHERE LOWER(product_name) LIKE '%{clean_model}%'",
            # Special handling for fold models
            f"SELECT store, product_name, price, url, '9' as relevanceScore FROM products WHERE LOWER(product_name) LIKE '%fold%' AND '{clean_model}' LIKE '%fold%'",
            # Special handling for flip models
            f"SELECT store, product_name, price, url, '9' as relevanceScore FROM products WHERE LOWER(product_name) LIKE '%flip%' AND '{clean_model}' LIKE '%flip%'",
            # Search keywords match
            f"SELECT store, product_name, price, url, '8' as relevanceScore FROM products WHERE LOWER(search_keywords) LIKE '%{clean_model}%'",
            # Brand + model combination
            f"SELECT store, product_name, price, url, '7' as relevanceScore FROM products WHERE LOWER(search_keywords) LIKE '%{clean_brand}%{clean_model}%'",
            # Brand match only (fallback)
            f"SELECT store, product_name, price, url, '6' as relevanceScore FROM products WHERE LOWER(search_keywords) LIKE '%{clean_brand}%' ORDER BY exact_model_match DESC, price ASC"
        ]
        
        products = []
        for i, query in enumerate(queries):
            try:
                cursor.execute(query)
                results = cursor.fetchall()
                print(f"  Query {i+1} found {len(results)} results")
                
                for row in results:
                    product = {
                        "store": row[0],
                        "productName": row[1],
                        "price": row[2],
                        "url": row[3],
                        "relevanceScore": int(row[4])
                    }
                    # Avoid duplicates
                    if not any(p['productName'] == product['productName'] for p in products):
                        products.append(product)
                        print(f"    ✅ Added: {product['productName']}")
                
                # If we found good matches, break early
                if len(products) >= 6:
                    break
                    
            except Exception as query_error:
                print(f"  Query {i+1} error: {query_error}")
                continue
        
        conn.close()
        
        if products:
            print(f"📊 Found {len(products)} products in database for {brand} {model}")
            return products
        else:
            print(f"📊 No products found, using fallback for {brand} {model}")
            return get_fallback_products(f"{brand} {model}")
            
    except Exception as e:
        print(f"Enhanced database search error: {e}")
        return get_fallback_products(f"{brand} {model}")
        
def get_products_from_database(search_term):
    """Get products from database when scraping fails"""
    try:
        conn = sqlite3.connect('products.db')
        cursor = conn.cursor()
        
        # Search in product names and keywords
        query = '''
            SELECT store, product_name, price, url, 'N/A' as relevanceScore 
            FROM products 
            WHERE product_name LIKE ? OR search_keywords LIKE ?
            ORDER BY exact_model_match DESC, price ASC
            LIMIT 6
        '''
        
        search_pattern = f'%{search_term}%'
        cursor.execute(query, (search_pattern, search_pattern))
        
        products = []
        for row in cursor.fetchall():
            products.append({
                "store": row[0],
                "productName": row[1],
                "price": row[2],
                "url": row[3],
                "relevanceScore": 8  # Default score for database results
            })
        
        conn.close()
        
        if products:
            print(f"📊 Found {len(products)} products in database for '{search_term}'")
        else:
            print(f"📊 No products found in database for '{search_term}'")
            products = get_fallback_products(search_term)
            
        return products
        
    except Exception as e:
        print(f"Database search error: {e}")
        return get_fallback_products(search_term)

def get_fallback_products(search_term):
    """Return fallback products when no results found"""
    search_lower = search_term.lower()
    
    # Enhanced fallback with specific model matching
    if any(word in search_lower for word in ['samsung', 'galaxy']):
        if 'fold' in search_lower:
            return [
                {"store": "Amazon", "productName": "Samsung Galaxy Z Fold5 (512GB, Phantom Black)", "price": 154999, "url": "https://www.amazon.in/s?k=samsung+z+fold5", "relevanceScore": 10},
                {"store": "Flipkart", "productName": "Samsung Galaxy Z Fold5 (256GB, Cream)", "price": 144999, "url": "https://www.flipkart.com/search?q=samsung+z+fold5", "relevanceScore": 9},
                {"store": "Amazon", "productName": "Samsung Galaxy Z Fold4 (256GB, Graygreen)", "price": 124999, "url": "https://www.amazon.in/s?k=samsung+z+fold4", "relevanceScore": 8}
            ]
        elif 'flip' in search_lower:
            return [
                {"store": "Amazon", "productName": "Samsung Galaxy Z Flip5 (256GB, Mint)", "price": 89999, "url": "https://www.amazon.in/s?k=samsung+z+flip5", "relevanceScore": 10},
                {"store": "Flipkart", "productName": "Samsung Galaxy Z Flip5 (512GB, Graphite)", "price": 99999, "url": "https://www.flipkart.com/search?q=samsung+z+flip5", "relevanceScore": 9},
                {"store": "Amazon", "productName": "Samsung Galaxy Z Flip4 (256GB, Blue)", "price": 74999, "url": "https://www.amazon.in/s?k=samsung+z+flip4", "relevanceScore": 8}
            ]
        else:
            return [
                {"store": "Amazon", "productName": "Samsung Galaxy S25 Ultra (512GB, Phantom Black)", "price": 139999, "url": "https://www.amazon.in/s?k=samsung+s25+ultra", "relevanceScore": 10},
                {"store": "Flipkart", "productName": "Samsung Galaxy S24 Ultra (256GB, Titanium Violet)", "price": 124999, "url": "https://www.flipkart.com/search?q=samsung+s24+ultra", "relevanceScore": 9},
                {"store": "Amazon", "productName": "Samsung Galaxy A55 5G (128GB, Awesome Iceblue)", "price": 34999, "url": "https://www.amazon.in/s?k=samsung+a55", "relevanceScore": 8}
            ]
    
    elif any(word in search_lower for word in ['apple', 'iphone']):
        return [
            {"store": "Amazon", "productName": "Apple iPhone 16 Pro (256GB, Space Black)", "price": 134999, "url": "https://www.amazon.in/s?k=iphone+16+pro", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Apple iPhone 16 (128GB, Pink)", "price": 79999, "url": "https://www.flipkart.com/search?q=iphone+16", "relevanceScore": 9},
            {"store": "Amazon", "productName": "Apple iPhone 15 (128GB, Black)", "price": 69999, "url": "https://www.amazon.in/s?k=iphone+15", "relevanceScore": 8}
        ]
    
    elif any(word in search_lower for word in ['oneplus']):
        return [
            {"store": "Amazon", "productName": "OnePlus 12 (256GB, Silky Black)", "price": 69999, "url": "https://www.amazon.in/s?k=oneplus+12", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "OnePlus 12R (128GB, Iron Gray)", "price": 39999, "url": "https://www.flipkart.com/search?q=oneplus+12r", "relevanceScore": 9},
            {"store": "Amazon", "productName": "OnePlus Nord CE 3 (128GB, Aqua Surge)", "price": 24999, "url": "https://www.amazon.in/s?k=oneplus+nord+ce+3", "relevanceScore": 8}
        ]
    
    elif any(word in search_lower for word in ['xiaomi', 'redmi', 'poco']):
        return [
            {"store": "Amazon", "productName": "Xiaomi 13 Pro (256GB, Ceramic White)", "price": 79999, "url": "https://www.amazon.in/s?k=xiaomi+13+pro", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Redmi Note 13 Pro+ (256GB, Fusion Purple)", "price": 33999, "url": "https://www.flipkart.com/search?q=redmi+note+13+pro+plus", "relevanceScore": 9},
            {"store": "Amazon", "productName": "Poco X6 Pro (256GB, Racing Gray)", "price": 26999, "url": "https://www.amazon.in/s?k=poco+x6+pro", "relevanceScore": 8}
        ]
    
    elif any(word in search_lower for word in ['sony']):
        return [
            {"store": "Amazon", "productName": "Sony WH-1000XM5 Wireless Headphones (Black)", "price": 29990, "url": "https://www.amazon.in/s?k=sony+wh1000xm5", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Sony PlayStation 5 (Standard Edition)", "price": 50990, "url": "https://www.flipkart.com/search?q=playstation+5", "relevanceScore": 9},
            {"store": "Amazon", "productName": "Sony Bravia 55-inch 4K Android TV", "price": 74990, "url": "https://www.amazon.in/s?k=sony+bravia+4k", "relevanceScore": 8}
        ]
    
    else:
        # Generic fallback
        return [
            {"store": "Amazon", "productName": "Apple iPhone 16 Pro (256GB, Space Black)", "price": 134999, "url": "https://www.amazon.in/s?k=iphone+16+pro", "relevanceScore": 10},
            {"store": "Flipkart", "productName": "Samsung Galaxy S25 Ultra (512GB, Phantom Black)", "price": 139999, "url": "https://www.flipkart.com/search?q=samsung+s25+ultra", "relevanceScore": 9},
            {"store": "Amazon", "productName": "OnePlus 12 (256GB, Silky Black)", "price": 69999, "url": "https://www.amazon.in/s?k=oneplus+12", "relevanceScore": 8}
        ]

@app.route("/")
def serve_index():
    try:
        return send_file("index.html")
    except Exception as e:
        return jsonify({"error": "Could not load interface", "details": str(e)}), 500

# Brand recognition endpoint
@app.route("/recognize-brand", methods=["POST"])
def recognize_brand():
    """Enhanced brand recognition with multiple methods"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        f = request.files["image"]
        if f.filename == '':
            return jsonify({"error": "No image selected"}), 400

        if not allowed_file(f.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save uploaded file
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        detected_brand = None
        detection_method = "unknown"
        confidence = 0
        
        print(f"\n🔍 Starting enhanced brand detection for: {filename}")
        
        # Method 1: Try enhanced multi-method detection
        try:
            detected_brand, detection_method, confidence = brand_detector.detect_brand(filepath, filename)
            print(f"✅ Enhanced detection: {detected_brand} (method: {detection_method}, confidence: {confidence})")
        except Exception as e:
            print(f"Enhanced detection failed: {e}")
        
        # Method 2: Fallback to ML model (if available)
        if (not detected_brand or detected_brand == "Unknown" or confidence < 0.5) and ML_MODEL_LOADED:
            try:
                ml_brand = detect_brand_ml(filepath)
                if ml_brand and ml_brand != "Unknown":
                    detected_brand = ml_brand
                    detection_method = "ml_model"
                    confidence = 0.7
                    print(f"🤖 ML fallback: {detected_brand}")
            except Exception as e:
                print(f"ML detection failed: {e}")
        
        # Method 3: Final filename fallback
        if not detected_brand or detected_brand == "Unknown":
            detected_brand = brand_detector.fallback_detection(filename)
            detection_method = "filename_fallback"
            confidence = 0.3
            print(f"📁 Final fallback: {detected_brand}")

        # Get suggestions with debugging
        print(f"🎯 Getting suggestions for brand: {detected_brand}")
        suggestions = get_brand_suggestions(detected_brand)
        print(f"📋 Suggestions found: {suggestions}")

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        # Prepare response message based on confidence
        if confidence >= 0.7:
            message = f"✅ Confidently detected {detected_brand} brand"
        elif confidence >= 0.5:
            message = f"🟡 Likely detected {detected_brand} brand"
        else:
            message = f"🟠 Possibly detected {detected_brand} brand (low confidence)"
        
        message += f". Please verify and select a model below."
        
        return jsonify({
            "detected_brand": detected_brand,
            "suggestions": suggestions,
            "detection_method": detection_method,
            "confidence": confidence,
            "message": message,
            "ml_used": detection_method == "ml_model"
        })
        
    except Exception as e:
        print(f"💥 Brand recognition error: {e}")
        return jsonify({"error": f"Brand recognition failed: {str(e)}"}), 500
    
@app.route("/compare", methods=["POST"])
def compare():
    try:
        data = request.get_json()
        if not data or "productName" not in data:
            return jsonify({"error": "Missing productName"}), 400

        product_name = data["productName"].strip()
        if not product_name:
            return jsonify({"error": "Empty productName"}), 400

        print(f"🎯 SMART SEARCH for: '{product_name}'")
        
        # Use enhanced search with better filtering
        products = search_products_real_time(product_name)
        
        print(f"✅ Returning {len(products)} RELEVANT products for '{product_name}'")
        for p in products:
            print(f"   📦 {p['productName']} - ₹{p['price']}")
        
        return jsonify({
            "products": products,
            "search_term": product_name,
            "source": "smart_search"
        })
        
    except Exception as e:
        print(f"💥 Compare error: {e}")
        return jsonify({
            "products": get_smart_fallback(product_name),
            "search_term": product_name,
            "source": "error_fallback"
        })

# New endpoint for direct model-based search
@app.route("/search-model", methods=["POST"])
def search_model():
    """Enhanced search for specific model with brand context"""
    try:
        data = request.get_json()
        brand = data.get("brand", "").strip()
        model = data.get("model", "").strip()
        
        if not model:
            return jsonify({"error": "Missing model name"}), 400
        
        print(f"🔍 ENHANCED MODEL SEARCH: Brand='{brand}', Model='{model}'")
        
        # Use enhanced search
        products = search_products_by_model(brand, model)
        
        return jsonify({
            "products": products,
            "search_query": f"{brand} {model}",
            "brand": brand,
            "model": model,
            "source": "enhanced_model_search"
        })
        
    except Exception as e:
        print(f"💥 Enhanced model search error: {e}")
        return jsonify({"error": f"Model search failed: {str(e)}"}), 500
    
# Additional endpoints
@app.route("/brand-suggestions/<brand>", methods=["GET"])
def get_brand_suggestions_endpoint(brand):
    suggestions = get_brand_suggestions(brand.capitalize())
    return jsonify({"brand": brand, "suggestions": suggestions})

@app.route("/reinit-db", methods=["POST"])
def reinit_db():
    """Reinitialize database with sample data"""
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS products")
    conn.commit()
    conn.close()
    
    init_database()
    return jsonify({"message": "Database reinitialized successfully"})

@app.route("/api/status")
def api_status():
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products")
    product_count = cursor.fetchone()[0]
    conn.close()
    
    return jsonify({
        "message": "SmartShop API is running with Enhanced Brand Recognition", 
        "statistics": {
            "cached_products": product_count,
            "supported_brands": len(KNOWN_BRANDS),
            "ml_model_loaded": ML_MODEL_LOADED
        },
        "supported_brands": list(KNOWN_BRANDS.keys()),
        "ml_enabled": ML_MODEL_LOADED,
        "features": [
            "Multi-method Brand Detection", 
            "Real-time Scraping", 
            "Database Fallback",
            "Model Suggestions",
            "Price Comparison"
        ]
    })

# Health check endpoint
@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "ml_model": "loaded" if ML_MODEL_LOADED else "not_loaded",
        "database": "connected",
        "scraping": "available"
    })

if __name__ == "__main__":
    app.run(debug = False, port="Your port")