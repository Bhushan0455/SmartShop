# collect_data.py - Data collection for brand detection
import os
import requests
from bs4 import BeautifulSoup
import time
import random
from PIL import Image
from io import BytesIO
import json

class DataCollector:
    def __init__(self):
        self.brands = {
            'samsung': ['samsung galaxy', 'samsung phone', 'samsung smartphone'],
            'apple': ['iphone', 'apple phone', 'iphone pro'],
            'oneplus': ['oneplus phone', 'oneplus smartphone'],
            'xiaomi': ['xiaomi phone', 'redmi phone', 'poco phone'],
            'sony': ['sony xperia', 'sony phone'],
            'google': ['google pixel', 'pixel phone'],
            'realme': ['realme phone'],
            'oppo': ['oppo phone'],
            'vivo': ['vivo phone'],
            'nothing': ['nothing phone'],
            'motorola': ['motorola phone', 'moto phone']
        }
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Create directories for each brand
        for brand in self.brands.keys():
            os.makedirs(f'training_data/{brand}', exist_ok=True)
    
    def download_images_google(self, query, brand, max_images=50):
        """Download images using Google search (educational purpose)"""
        print(f"📸 Downloading {max_images} images for {brand} - {query}")
        
        # Note: In production, you would use proper image APIs or datasets
        # This is a simplified version for demonstration
        
        try:
            # For educational purposes, we'll create synthetic data
            # In real scenario, you'd use proper image sources
            self.create_synthetic_images(brand, max_images)
            
        except Exception as e:
            print(f"Error downloading images for {brand}: {e}")
            self.create_synthetic_images(brand, max_images)
    
    def create_synthetic_images(self, brand, num_images=20):
        """Create synthetic training images for demonstration"""
        print(f"🎨 Creating synthetic images for {brand}")
        
        # Brand color schemes
        brand_colors = {
            'samsung': [(0, 112, 186), (0, 0, 0), (255, 255, 255)],  # Blue, Black, White
            'apple': [(128, 128, 128), (255, 255, 255), (0, 0, 0)],   # Gray, White, Black
            'oneplus': [(233, 0, 0), (0, 0, 0), (255, 255, 255)],     # Red, Black, White
            'xiaomi': [(255, 103, 31), (0, 0, 0), (255, 255, 255)],   # Orange, Black, White
            'sony': [(0, 0, 0), (0, 97, 173), (255, 255, 255)],       # Black, Blue, White
            'google': [(66, 133, 244), (52, 168, 83), (255, 255, 255)], # Blue, Green, White
            'realme': [(255, 215, 0), (0, 0, 0), (255, 255, 255)],    # Yellow, Black, White
            'oppo': [(0, 150, 136), (0, 0, 0), (255, 255, 255)],      # Green, Black, White
            'vivo': [(0, 114, 188), (0, 0, 0), (255, 255, 255)],      # Blue, Black, White
            'nothing': [(255, 255, 255), (0, 0, 0), (128, 128, 128)], # White, Black, Gray
            'motorola': [(0, 0, 0), (255, 255, 255), (0, 112, 186)]   # Black, White, Blue
        }
        
        colors = brand_colors.get(brand, [(0, 0, 0), (255, 255, 255)])
        
        for i in range(num_images):
            try:
                # Create image with brand colors
                img = self.generate_brand_image(colors, brand, i)
                
                # Save image
                filename = f'training_data/{brand}/{brand}_{i:04d}.jpg'
                img.save(filename, 'JPEG', quality=85)
                
            except Exception as e:
                print(f"Error creating image {i} for {brand}: {e}")
    
    def generate_brand_image(self, colors, brand, index):
        """Generate a synthetic brand image"""
        width, height = 224, 224
        img = Image.new('RGB', (width, height), color=colors[0])
        
        # Add some patterns based on brand
        if brand == 'apple':
            # Apple-like simple design
            for x in range(0, width, 20):
                for y in range(0, height, 20):
                    if (x + y) % 40 == 0:
                        img.putpixel((x, y), colors[1])
                        
        elif brand == 'samsung':
            # Samsung-like pattern
            for x in range(0, width, 15):
                img.putpixel((x, height//2), colors[1])
            for y in range(0, height, 15):
                img.putpixel((width//2, y), colors[1])
                
        elif brand == 'oneplus':
            # OnePlus red accent
            for i in range(50):
                x = random.randint(0, width-1)
                y = random.randint(0, height-1)
                img.putpixel((x, y), colors[1])
                
        # Add some random noise to make images more varied
        for _ in range(100):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            r, g, b = img.getpixel((x, y))
            img.putpixel((x, y)), (
                min(255, max(0, r + random.randint(-10, 10))),
                min(255, max(0, g + random.randint(-10, 10))),
                min(255, max(0, b + random.randint(-10, 10)))
            )
        
        return img
    
    def collect_all_data(self, images_per_brand=100):
        """Collect data for all brands"""
        print("🚀 Starting data collection...")
        
        for brand, queries in self.brands.items():
            print(f"\n📁 Processing {brand}...")
            for query in queries:
                self.download_images_google(query, brand, images_per_brand // len(queries))
            
            # Add some delay to be respectful
            time.sleep(1)
        
        print("\n✅ Data collection completed!")
        self.create_dataset_info()
    
    def create_dataset_info(self):
        """Create dataset information file"""
        dataset_info = {
            'total_images': 0,
            'brands': {},
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        for brand in self.brands.keys():
            brand_dir = f'training_data/{brand}'
            if os.path.exists(brand_dir):
                images = [f for f in os.listdir(brand_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                dataset_info['brands'][brand] = len(images)
                dataset_info['total_images'] += len(images)
        
        with open('training_data/dataset_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"📊 Dataset info: {dataset_info}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all_data(images_per_brand=50)