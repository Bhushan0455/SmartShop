🛍️ SmartShop – AI Powered Brand Detection & Smart Price Comparison
🚀 Problem Statement

Online shoppers often see products in real life (stores, ads, social media) but:

They don’t know the exact brand/model.

They spend time manually searching across multiple platforms.

They miss better deals on other websites.

This results in time loss, price confusion, and poor purchase decisions.

💡 Solution

SmartShop is an AI-powered web application that:

📸 Takes an uploaded product image

🧠 Uses Deep Learning (MobileNetV2) to detect the product brand

🔎 Automatically searches for that product on e-commerce platforms

💰 Displays price comparisons from Amazon and Flipkart

This enables instant brand identification + smarter price decisions in one step.

🧠 How AI is Used

Model: MobileNetV2 (Transfer Learning)

Framework: TensorFlow / Keras

Task: Image Classification

Output: Brand prediction with confidence score

Instead of manually searching, the AI model identifies the brand directly from the uploaded image.

📊 Sample Results & Metrics

(Replace with your actual numbers if available)

📈 Model Accuracy: ~88–92% on validation dataset

🏷️ Number of Brands Supported: 10+ electronics brands

⚡ Average Prediction Time: < 2 seconds

📦 Real-time price comparison from Amazon & Flipkart

Example Prediction

Input: Image of a Samsung TV
Output:

Predicted Brand: Samsung

Confidence Score: 91.4%

Price Results:

Amazon: ₹42,999

Flipkart: ₹41,499

🛠️ Tech Stack

Backend: Python (Flask)

Frontend: HTML, CSS, JavaScript

AI Model: MobileNetV2 (TensorFlow / Keras)

Database: SQLite

Web Scraping: Requests / BeautifulSoup (if used)

🏗️ System Architecture

User Uploads Image
⬇
Flask Backend Receives Image
⬇
Image Preprocessing
⬇
MobileNetV2 Model Prediction
⬇
Brand Identified
⬇
Price Scraper Fetches Deals
⬇
Results Displayed to User

🌍 Why It Matters

SmartShop bridges the gap between computer vision AI and real-world shopping utility.

It demonstrates how AI can:

Save time for consumers

Improve purchasing decisions

Provide instant brand intelligence

Combine computer vision + web automation in one solution

This project showcases a practical application of deep learning beyond academic datasets.

▶️ Deployment Instructions (Run Locally)
1️⃣ Clone the Repository
git clone https://github.com/Bhushan0455/SmartShop.git
cd SmartShop

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python app.py

5️⃣ Open in Browser

Go to:

http://127.0.0.1:5000/
