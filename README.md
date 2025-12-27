🛒 SmartShop: AI Brand Recognition & Price Comparison
SmartShop is an intelligent web application that identifies electronics brands from images and instantly fetches the best deals from major e-commerce platforms like Amazon and Flipkart.
It combines Deep Learning (MobileNetV2), metadata analysis, and real-time web scraping to provide a seamless shopping experience.

✨ Features
Multi-Method Brand Detection: Uses a 3-layer detection system:
Deep Learning: A CNN model (MobileNetV2) trained to recognize brand aesthetics.
Metadata Analysis: Scans filenames and image data for brand keywords.
Color Profiling: Analyzes dominant image colors to match brand identities (e.g., Samsung Blue, OnePlus Red).
Live Web Scraping: Scrapes Amazon and Flipkart in real-time for live prices.
Smart Database: Uses SQLite3 to cache results and provide instant fallbacks.
Responsive UI: A clean, modern dashboard for uploading images and viewing price comparisons.

🛠️ Tech Stack
Backend: Python (Flask)
Frontend: HTML5, CSS3, JavaScript
AI/ML: TensorFlow, Keras, Pillow, Scikit-learn
Database: SQLite3
Scraping: BeautifulSoup4, Requests

📦 Installation
Clone the Repository:
git clone https://github.com/YOUR_USERNAME/smartshop.git
cd smartshop

Install Dependencies
pip install -r requirements.txt

Run the Application
python app.py

Access the Web Interface Open your browser and navigate to your corresponding port.

🧠 How the AI Works
The project uses Transfer Learning with the MobileNetV2 architecture.
Training: The train_model.py script processes images in the real_training_data folder. If no data is present, it can generate synthetic training samples to get the model started.
Inferenc: When you upload an image, the model calculates a confidence score for each brand (Apple, Samsung, OnePlus, Xiaomi, Sony).
Fallback: If the image is unclear, the system falls back to keyword detection in the filename to ensure the user still gets price results.

📂 Project Structure
app.py: The main Flask server and API.
scraper.py: Logic for extracting data from Amazon and Flipkart.
enhanced_brand_detector.py: The logic for multi-method brand identification.
Products_data.py: A comprehensive dataset of products used to seed the local database.
train_model.py: Script to train or update the AI model.
improved_brand_model.h5: The pre-trained weights for the AI.

🛡️ Security & Privacy
Local Data: The project is configured to ignore local databases (*.db) and raw training datasets via .gitignore to keep the repository lightweight and private.
Ethical Scraping: The scraper includes headers to mimic a browser and identifies as a tool for personal use.
