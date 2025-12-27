import os
import shutil

def organize_training_data():
    """Organize and fix the training data structure"""
    
    # Define correct folder structure
    correct_structure = {
        'apple': 'apple',
        'Apple': 'apple',
        'samsung': 'samsung', 
        'Samsung': 'samsung',
        'oneplus': 'oneplus',
        'Oneplus': 'oneplus',
        'sony': 'sony',
        'Sony': 'sony'
    }
    
    base_dir = 'real_training_data'
    
    # Create all required brand folders
    required_brands = ['apple', 'samsung', 'oneplus', 'sony', 'xiaomi']
    for brand in required_brands:
        os.makedirs(os.path.join(base_dir, brand), exist_ok=True)
    
    # Move and rename files to correct folders
    for old_brand, new_brand in correct_structure.items():
        old_path = os.path.join(base_dir, old_brand)
        new_path = os.path.join(base_dir, new_brand)
        
        if os.path.exists(old_path) and old_brand != new_brand:
            print(f"Moving {old_brand} -> {new_brand}")
            
            # Move all files from old folder to new folder
            for filename in os.listdir(old_path):
                old_file_path = os.path.join(old_path, filename)
                new_file_path = os.path.join(new_path, filename)
                
                if os.path.isfile(old_file_path):
                    shutil.move(old_file_path, new_file_path)
                    print(f"  Moved: {filename}")
            
            # Remove old empty folder
            try:
                os.rmdir(old_path)
            except:
                pass
    
    # Handle subfolders (like Macbook inside apple)
    for brand in required_brands:
        brand_path = os.path.join(base_dir, brand)
        
        for item in os.listdir(brand_path):
            item_path = os.path.join(brand_path, item)
            
            if os.path.isdir(item_path):
                print(f"Moving files from subfolder: {item_path}")
                
                # Move files from subfolder to main brand folder
                for filename in os.listdir(item_path):
                    old_file_path = os.path.join(item_path, filename)
                    new_file_path = os.path.join(brand_path, filename)
                    
                    if os.path.isfile(old_file_path):
                        shutil.move(old_file_path, new_file_path)
                        print(f"  Moved: {filename}")
                
                # Remove empty subfolder
                try:
                    os.rmdir(item_path)
                except:
                    pass
    
    # Standardize file extensions to .jpg
    for brand in required_brands:
        brand_path = os.path.join(base_dir, brand)
        
        for filename in os.listdir(brand_path):
            if filename.lower().endswith(('.jpeg', '.png')):
                old_path = os.path.join(brand_path, filename)
                name_without_ext = os.path.splitext(filename)[0]
                new_path = os.path.join(brand_path, name_without_ext + '.jpg')
                
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {name_without_ext + '.jpg'}")
    
    print("✅ Data organization completed!")

def check_training_data():
    """Check if training data meets requirements"""
    base_dir = 'real_training_data'
    required_brands = ['apple', 'samsung', 'oneplus', 'sony', 'xiaomi']
    
    print("📊 Training Data Status:")
    print("-" * 40)
    
    for brand in required_brands:
        brand_path = os.path.join(base_dir, brand)
        
        if os.path.exists(brand_path):
            images = [f for f in os.listdir(brand_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"✅ {brand}: {len(images)} images")
            
            if len(images) < 10:
                print(f"   ⚠️  Need more images for {brand} (minimum 10 recommended)")
        else:
            print(f"❌ {brand}: Folder missing!")
    
    print("-" * 40)

if __name__ == "__main__":
    organize_training_data()
    check_training_data()