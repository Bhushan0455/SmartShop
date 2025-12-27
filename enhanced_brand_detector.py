# enhanced_brand_detector.py - Multi-method brand detection
import os
import re
from PIL import Image
import numpy as np

class EnhancedBrandDetector:
    def __init__(self):
        self.brand_keywords = {
            'apple': ['iphone', 'ipad', 'macbook', 'apple', 'ios', 'applestore', 'airpods'],
            'samsung': ['samsung', 'galaxy', 'note', 'z fold', 'z flip', 'android', 'one ui'],
            'oneplus': ['oneplus', 'nord', 'oxygenos', 'alert slider'],
            'xiaomi': ['xiaomi', 'mi ', 'redmi', 'poco', 'miui'],
            'sony': ['sony', 'xperia', 'playstation', 'bravia', 'alpha']
        }
        
        self.brand_colors = {
            'apple': [(50, 50, 50), (200, 200, 200), (255, 255, 255)],  # Black, Gray, White
            'samsung': [(0, 112, 186), (0, 0, 0), (255, 255, 255)],     # Blue, Black, White
            'oneplus': [(233, 0, 0), (0, 0, 0), (255, 255, 255)],       # Red, Black, White
            'xiaomi': [(255, 103, 31), (0, 0, 0), (255, 255, 255)],     # Orange, Black, White
            'sony': [(0, 0, 0), (0, 97, 173), (255, 255, 255)]          # Black, Blue, White
        }
    
    def detect_from_filename(self, filename):
        """Detect brand from filename with enhanced matching"""
        filename_lower = filename.lower()
        
        # Remove file extension and common words
        clean_name = re.sub(r'\.(jpg|jpeg|png|gif)$', '', filename_lower)
        clean_name = re.sub(r'[_-]', ' ', clean_name)
        
        print(f"🔍 Analyzing filename: {clean_name}")
        
        # Check for exact brand matches first
        for brand, keywords in self.brand_keywords.items():
            for keyword in keywords:
                if keyword in clean_name:
                    confidence = 0.9 if keyword in ['apple', 'samsung', 'oneplus', 'xiaomi', 'sony'] else 0.7
                    print(f"✅ Filename match: {brand} (confidence: {confidence})")
                    return brand.capitalize(), confidence
        
        return None, 0
    
    def detect_from_image_analysis(self, image_path):
        """Enhanced visual analysis of images"""
        try:
            img = Image.open(image_path)
            img = img.resize((100, 100))
            img_array = np.array(img)
            
            # Calculate dominant colors
            avg_color = np.mean(img_array, axis=(0, 1))
            std_color = np.std(img_array, axis=(0, 1))
            
            print(f"🎨 Image analysis - Avg RGB: {avg_color}, Std: {std_color}")
            
            # Check for Apple characteristics (minimalist, dark/light)
            if std_color.mean() < 30:  # Low variation (minimalist design)
                if avg_color[0] < 100 and avg_color[1] < 100 and avg_color[2] < 100:
                    print("✅ Visual match: Apple (dark minimalist)")
                    return "Apple", 0.6
                elif avg_color[0] > 200 and avg_color[1] > 200 and avg_color[2] > 200:
                    print("✅ Visual match: Apple (light minimalist)")
                    return "Apple", 0.6
            
            # Check for Samsung (often blue tones)
            if avg_color[2] > avg_color[0] + 20 and avg_color[2] > avg_color[1] + 20:
                print("✅ Visual match: Samsung (blue dominant)")
                return "Samsung", 0.5
            
            # Check for OnePlus (red accents)
            if avg_color[0] > avg_color[1] + 30 and avg_color[0] > avg_color[2] + 30:
                print("✅ Visual match: OnePlus (red dominant)")
                return "OnePlus", 0.5
                
        except Exception as e:
            print(f"Visual analysis error: {e}")
        
        return None, 0
    
    def detect_from_metadata(self, image_path):
        """Extract brand info from image metadata"""
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if exif_data:
                # Check for camera make in EXIF data
                camera_make = exif_data.get(271)  # Make tag
                if camera_make:
                    camera_make_lower = camera_make.lower()
                    for brand in self.brand_keywords.keys():
                        if brand in camera_make_lower:
                            print(f"✅ EXIF match: {brand}")
                            return brand.capitalize(), 0.8
                            
        except Exception as e:
            print(f"Metadata analysis error: {e}")
        
        return None, 0
    
    def fallback_detection(self, filename):
        """Final fallback based on common patterns"""
        filename_lower = filename.lower()
        
        print(f"🔄 Using fallback detection for: {filename_lower}")
        
        # Common iPhone patterns
        iphone_indicators = ['iphone', 'ip', 'apple', 'ios']
        if any(indicator in filename_lower for indicator in iphone_indicators):
            print("📱 iPhone detected in fallback")
            return "Apple"
        
        # Common Samsung patterns
        samsung_indicators = ['galaxy', 'samsung', 's25', 's24', 's23', 'note']
        if any(indicator in filename_lower for indicator in samsung_indicators):
            print("📱 Samsung detected in fallback")
            return "Samsung"
        
        # Common OnePlus patterns
        oneplus_indicators = ['oneplus', 'op', 'nord']
        if any(indicator in filename_lower for indicator in oneplus_indicators):
            print("📱 OnePlus detected in fallback")
            return "OnePlus"
        
        # Common Xiaomi patterns
        xiaomi_indicators = ['xiaomi', 'mi ', 'redmi', 'poco']
        if any(indicator in filename_lower for indicator in xiaomi_indicators):
            print("📱 Xiaomi detected in fallback")
            return "Xiaomi"
        
        # Common Sony patterns
        sony_indicators = ['sony', 'xperia', 'playstation']
        if any(indicator in filename_lower for indicator in sony_indicators):
            print("📱 Sony detected in fallback")
            return "Sony"
        
        print("❓ No brand detected in fallback")
        return "Unknown"
    
    def detect_brand(self, image_path, filename):
        """Multi-method brand detection with confidence scoring"""
        print(f"\n🎯 Starting brand detection for: {filename}")
        
        methods = []
        confidences = []
        
        # Method 1: Filename analysis
        brand_filename, conf_filename = self.detect_from_filename(filename)
        if brand_filename:
            methods.append(("filename", brand_filename, conf_filename))
        
        # Method 2: Image analysis
        brand_visual, conf_visual = self.detect_from_image_analysis(image_path)
        if brand_visual:
            methods.append(("visual", brand_visual, conf_visual))
        
        # Method 3: Metadata analysis
        brand_meta, conf_meta = self.detect_from_metadata(image_path)
        if brand_meta:
            methods.append(("metadata", brand_meta, conf_meta))
        
        # Combine results
        if methods:
            # Sort by confidence
            methods.sort(key=lambda x: x[2], reverse=True)
            best_method, best_brand, best_confidence = methods[0]
            
            # Ensure consistent brand naming (capitalize properly)
            best_brand = best_brand.capitalize()
            if best_brand == "Oneplus":  # Fix common capitalization issue
                best_brand = "OnePlus"
            
            print(f"🏆 Best detection: {best_brand} via {best_method} (confidence: {best_confidence})")
            
            # Apply confidence threshold
            if best_confidence >= 0.4:
                return best_brand, best_method, best_confidence
            else:
                print("⚠️ Low confidence, using fallback")
                fallback_brand = self.fallback_detection(filename)
                # Fix capitalization in fallback too
                if fallback_brand == "Oneplus":
                    fallback_brand = "OnePlus"
                return fallback_brand, "fallback", 0.3
        
        # Final fallback
        print("🔄 No methods succeeded, using final fallback")
        fallback_brand = self.fallback_detection(filename)
        if fallback_brand == "Oneplus":
            fallback_brand = "OnePlus"
        return fallback_brand, "fallback", 0.2

# Global instance
brand_detector = EnhancedBrandDetector()