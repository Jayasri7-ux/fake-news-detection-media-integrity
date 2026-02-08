# Fake News Detection – Media Integrity

A machine-learning based web application that detects fake news and assesses media integrity using NLP techniques.  
The system supports both **text-based** and **URL-based** news verification with confidence scores and explanations.

---

## 🚀 Live Demo

🔗 **https://fake-news-detection-media-integrity.onrender.com**

> (Free-tier hosting – first load may take ~30 seconds)

---

## 🎯 Project Overview

Fake news and misinformation pose serious risks in today’s digital world.  
This project provides an **end-to-end solution** for detecting fake news using multiple machine learning models and a production-ready web interface.

### What the system provides:
- Fake/Real news classification
- Confidence score and risk level
- Explanation with important keywords
- URL-based article extraction
- PDF report generation
- REST API for integration
- Full ML training and evaluation pipeline

---

## ✨ Key Features

- ✅ Text-based news classification  
- ✅ URL-based news classification (automatic article extraction)  
- ✅ Confidence score and risk-level analysis  
- ✅ Keyword-based explanations  
- ✅ Downloadable PDF prediction reports  
- ✅ RESTful API support  
- ✅ Production deployment using Gunicorn  

---

## 🧠 Machine Learning Models

The system evaluates multiple models and selects the best-performing one:

| Model | Accuracy | F1 Score |
|-----|---------|----------|
| Logistic Regression | 84.32% | 0.909 |
| Random Forest | 85.12% | 0.917 |
| Gradient Boosting | **85.26%** | **0.917** |

**Best Model:** Gradient Boosting

---

## 🛠️ Tech Stack

### Backend & ML
- Python
- Flask
- Scikit-learn
- Pandas, NumPy
- NLTK

### NLP & Extraction
- TF-IDF Vectorization
- Newspaper3k
- BeautifulSoup
- Language detection

### Deployment
- Gunicorn
- Render (Cloud Hosting)
- GitHub (Version Control)

---

## 📁 Project Structure (Simplified)

fake_news_detection_media_integrity/
├── __pycache__/
│   ├── extract_notebook_code.cpython-313.pyc
│   └── predict.cpython-313.pyc
│
├── artifacts/
│   ├── models/
│   └── reports/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── extracted_scripts/
│
├── fake-news-env/
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
│
├── models/
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_data_quality_report.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_training.ipynb
│   ├── 07_model_evaluation.ipynb
│   ├── 08_model_comparison.ipynb
│   ├── 09_model_training_comparison.ipynb
│   ├── 10_eda_dashboard.ipynb
│   └── verify_feature_engineering.py
│
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── utils/
│   └── run_pipeline.py
│
├── templates/
│   └── index.html
│
├── visuals/
│   ├── plots/
│   └── wordclouds/
│
├── venv/
│
├── .gitignore
├── accuracy_plot.py
├── app.py
├── create_wordcloud.py
├── extract_notebook_code.py
├── main.py
├── model_comparison_report.md
├── predict.py
├── Procfile
├── README.md
└── requirements.txt

---

## 💻 How to Use

### 🔹 Web Interface
1. Open the live app  
   👉 https://fake-news-detection-media-integrity.onrender.com
2. Paste news text **or** a news article URL
3. Click **Analyze**
4. View prediction, confidence, and explanation
5. Download PDF report (optional)

---

### 🔹 REST API

**Endpoint:** `POST /predict`

**Example request:**
```json
{
  "text": "Breaking news: Scientists discover new energy source",
  "is_url": false
}
Run Locally
pip install -r requirements.txt
python app.py
Open:
http://127.0.0.1:5000
Full ML Pipeline
python main.py
Pipeline includes:

Data collection

Data cleaning

Feature engineering

Model training

Model evaluation

Model comparison
Deployment

Deployed on Render

Production server: Gunicorn

Source control: GitHub

Live URL:
🔗 https://fake-news-detection-media-integrity.onrender.com