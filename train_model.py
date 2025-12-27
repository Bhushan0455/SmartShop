# fixed_improved_trainer.py - Fixed version without learning_rate error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os
import json
from PIL import Image
from sklearn.model_selection import train_test_split

class ImprovedBrandClassifier:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.brands = ['samsung', 'apple', 'oneplus', 'xiaomi', 'sony']
        self.model = None
        self.history = None
        
    def load_real_dataset(self):
        """Load the real/synthetic dataset"""
        print("📁 Loading real dataset...")
        
        X = []
        y = []
        
        for brand_idx, brand in enumerate(self.brands):
            brand_dir = f'real_training_data/{brand}'
            if not os.path.exists(brand_dir):
                print(f"⚠️ Directory {brand_dir} not found, creating synthetic data...")
                self.create_realistic_synthetic_dataset()
                break
                
        # Load images after creating dataset
        for brand_idx, brand in enumerate(self.brands):
            brand_dir = f'real_training_data/{brand}'
            if os.path.exists(brand_dir):
                image_files = [f for f in os.listdir(brand_dir) 
                              if f.endswith(('.jpg', '.jpeg', '.png'))]
                
                print(f"📸 Loading {len(image_files)} images for {brand}")
                
                for image_file in image_files:
                    try:
                        img_path = os.path.join(brand_dir, image_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.image_size)
                        img_array = np.array(img) / 255.0
                        
                        X.append(img_array)
                        y.append(brand_idx)
                        
                    except Exception as e:
                        print(f"Error loading {image_file}: {e}")
        
        if len(X) == 0:
            raise ValueError("❌ No training data found!")
            
        X = np.array(X)
        y = keras.utils.to_categorical(y, len(self.brands))
        
        print(f"✅ Loaded {len(X)} images across {len(self.brands)} brands")
        return X, y
    
    def create_realistic_synthetic_dataset(self):
        """Create a realistic synthetic dataset"""
        print("🎨 Creating realistic synthetic dataset...")
        
        # Create directories
        for brand in self.brands:
            os.makedirs(f'real_training_data/{brand}', exist_ok=True)
        
        # Create synthetic images for each brand
        for brand in self.brands:
            self.create_brand_images(brand, 50)
    
    def create_brand_images(self, brand, count=50):
        """Create realistic images for a specific brand"""
        print(f"  Creating {count} images for {brand}...")
        
        brand_colors = {
            'samsung': [(0, 112, 186), (0, 0, 0), (255, 255, 255)],
            'apple': [(128, 128, 128), (255, 255, 255), (0, 0, 0)],
            'oneplus': [(233, 0, 0), (0, 0, 0), (255, 255, 255)],
            'xiaomi': [(255, 103, 31), (0, 0, 0), (255, 255, 255)],
            'sony': [(0, 0, 0), (0, 97, 173), (255, 255, 255)]
        }
        
        colors = brand_colors.get(brand, [(0, 0, 0), (255, 255, 255)])
        
        for i in range(count):
            try:
                img = self.create_realistic_phone_image(colors, brand, i)
                filename = f"real_training_data/{brand}/{brand}_{i:03d}.jpg"
                img.save(filename, 'JPEG', quality=90)
                
            except Exception as e:
                print(f"Error creating image {i} for {brand}: {e}")
    
    def create_realistic_phone_image(self, colors, brand, index):
        """Create more realistic phone images"""
        width, height = 224, 224
        base_color = colors[index % len(colors)]
        
        # Create base image
        img = Image.new('RGB', (width, height), color=base_color)
        
        # Add realistic phone elements
        self.add_phone_elements(img, brand, index)
        
        return img
    
    def add_phone_elements(self, img, brand, index):
        """Add realistic phone design elements"""
        width, height = img.size
        
        # Add screen (different for each brand)
        screen_color = (20, 20, 20)  # Dark screen
        screen_margin = 15
        
        for x in range(screen_margin, width - screen_margin):
            for y in range(screen_margin + 20, height - screen_margin - 20):
                # Add some screen content variation
                if (x + y) % 10 == 0:
                    brightness = np.random.randint(30, 100)
                    img.putpixel((x, y), (brightness, brightness, brightness))
                else:
                    img.putpixel((x, y), screen_color)
        
        # Brand-specific elements
        if brand == 'apple':
            # iPhone notch
            for x in range(width//3, 2*width//3):
                for y in range(screen_margin, screen_margin + 10):
                    img.putpixel((x, y), (0, 0, 0))
        
        elif brand == 'samsung':
            # Samsung hole-punch
            center_x, center_y = width//2, screen_margin + 15
            for x in range(center_x-5, center_x+5):
                for y in range(center_y-5, center_y+5):
                    if (x-center_x)**2 + (y-center_y)**2 <= 25:
                        img.putpixel((x, y), (0, 0, 0))
        
        elif brand == 'oneplus':
            # OnePlus alert slider
            for x in range(width-15, width-5):
                for y in range(height//3, 2*height//3):
                    img.putpixel((x, y), (233, 0, 0))
    
    def build_improved_model(self):
        """Build an improved model using transfer learning"""
        print("🤖 Building improved model with transfer learning...")
        
        # Use MobileNetV2 as base (pre-trained on ImageNet)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(len(self.brands), activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✅ Improved model built successfully!")
        return model
    
    def train_improved_model(self, epochs=30, batch_size=16):
        """Train the improved model (simplified without fine-tuning)"""
        print("🚀 Starting improved model training...")
        
        # Load data
        X, y = self.load_real_dataset()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        print(f"📊 Training set: {X_train.shape[0]} images")
        print(f"📊 Validation set: {X_val.shape[0]} images")
        
        # Build model
        self.build_improved_model()
        
        # Enhanced callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'improved_brand_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Enhanced data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Train model
        print("🎯 Starting training...")
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('improved_brand_model.h5')
        
        # Save class names
        with open('improved_class_names.json', 'w') as f:
            json.dump(self.brands, f)
        
        # Evaluate final model
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Training completed! Final Validation Accuracy: {val_accuracy:.4f}")
        
        return self.history
    
    def predict_brand(self, image_path):
        """Predict brand from image with confidence"""
        if self.model is None:
            try:
                self.model = keras.models.load_model('improved_brand_model.h5')
                print("✅ Loaded improved pre-trained model")
            except:
                print("❌ No improved model found")
                return None, 0.0
        
        try:
            # Preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.image_size)
            img_array = np.array(img) / 255.0
            
            # Handle different image formats
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            brand_name = self.brands[predicted_class].capitalize()
            
            # Print top 3 predictions for debugging
            top_3 = np.argsort(predictions[0])[-3:][::-1]
            print("🔍 Top 3 predictions:")
            for i, idx in enumerate(top_3):
                print(f"  {i+1}. {self.brands[idx].capitalize()}: {predictions[0][idx]:.2f}")
            
            print(f"🎯 Final prediction: {brand_name} (confidence: {confidence:.2f})")
            return brand_name, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0.0

# Global functions for app integration
def initialize_ml_model():
    """Initialize the ML model - called by app.py"""
    print("🤖 Initializing Improved ML model...")
    
    # Check if model exists
    if os.path.exists('improved_brand_model.h5'):
        print("✅ Improved pre-trained model found!")
        return True
    else:
        print("⚠️ No improved model found. Please train the model first.")
        return False

def detect_brand_ml(image_path):
    """Detect brand from image using improved ML - called by app.py"""
    classifier = ImprovedBrandClassifier()
    brand, confidence = classifier.predict_brand(image_path)
    
    # Only return if confidence is reasonable
    if brand and confidence > 0.3:
        return brand
    return None

def train_model_now():
    """Train the model immediately"""
    print("🎯 Training Improved Brand Detection Model")
    print("=" * 50)
    
    # Create directories
    os.makedirs('real_training_data', exist_ok=True)
    
    # Train model
    classifier = ImprovedBrandClassifier()
    
    try:
        # Train the improved model
        history = classifier.train_improved_model(epochs=30, batch_size=16)
        
        print(f"\n🎉 Improved training completed!")
        print(f"💾 Model saved to: improved_brand_model.h5")
        print(f"🏷️ Class names saved to: improved_class_names.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    train_model_now()