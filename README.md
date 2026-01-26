# Fake News Detection Media Integrity

A Machine Learning–based Fake News Detection system that analyzes news text and classifies it as **Fake** or **Real**, with data processing, visualization, model evaluation, and Flask-based deployment.

---

## 📖 Project Overview

The rapid spread of fake news on digital platforms affects public trust and decision-making.  
This project aims to detect fake news articles using Machine Learning techniques by analyzing textual content and identifying misleading patterns.

The system includes:
- Data collection and cleaning
- Feature extraction using TF-IDF
- Model training and evaluation
- Visualization of insights
- Flask web application for prediction

---

## 🧰 Technologies Used

- Python  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- Flask  
- wordcloud  
- langdetect  
- Git & GitHub  

---

## 📁 Project Structure

fake-news-detection-media-integrity/
├── app/ # Flask application
├── artifacts/ # Saved models and reports
├── data/ # Raw and processed datasets
├── notebooks/ # Jupyter notebooks
├── src/ # Core ML and preprocessing modules
├── visuals/ # Generated plots and wordclouds
│ ├── plots/
│ └── wordclouds/
├── accuracy_plot.py
├── create_wordcloud.py
├── inspect_data.py
├── model_comparison_report.md
├── requirements.txt
└── README.md

---

## 📊 Visualizations

- Word Cloud showing frequently occurring words in fake news  
- Accuracy comparison plot for trained models  

---

## 🤖 Machine Learning Approach

- Text preprocessing and normalization  
- TF-IDF vectorization  
- Logistic Regression model  
- Evaluation using accuracy metrics  

---

## 🌐 Flask Application

The Flask app allows users to input news text and receive predictions indicating whether the news is **Fake** or **Real**.

---

## ▶️ How to Run the Project

### Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/fake_news_detection_media_integrity.git
cd fake_news_detection_media_integrity
Step 2: Create and activate virtual environment (Windows)
python -m venv fake-news-env
fake-news-env\Scripts\activate
Step 3: Install required dependencies
pip install -r requirements.txt
Step 4: Run the Flask application
cd app
python app.py
Step 5: Open the application in browser
http://127.0.0.1:5000/

