# IntegrityAI – Advanced Fake News Detection & Real-time Analytics

IntegrityAI is a professional-grade, machine-learning based web application designed to detect misinformation and assess media integrity. Featuring a modern **Glassmorphism dashboard**, the system provides deep insights through sentiment analysis, domain trust scoring, and interactive visualizations.

---

## 🚀 Live Demo

🔗 **[fake-news-detection-media-integrity.onrender.com](https://fake-news-detection-media-integrity.onrender.com)**

> [!NOTE]
> This app is hosted on a free-tier server. The first load may take ~30 seconds as the service wakes up.

---

## 🎯 Project Overview

IntegrityAI provides an **end-to-end solution** for verifying news content using a hybrid logic approach that combines rule-based heuristics with advanced Machine Learning models.

### What the system provides:
- **Real-time Classification**: Instantly identify if a story is "Real" or "Fake".
- **Multilingual Support**: Analyze and view results in **English**, **Hindi (हिंदी)**, and **Telugu (తెలుగు)**.
- **Bi-directional Translation**: Automatically translates non-English news for analysis and displays extracted content in your selected language.
- **Deep Analytics**: Sentiment profile, word metrics, and domain trustworthiness assessment.
- **PDF Reports**: Generate and download professional analysis reports with charts.

---

## 📊 Five Specialized Dashboards

1.  **🔍 Main Prediction**: The core analyzer. Supports direct text input and URL extraction with a dynamic confidence meter.
2.  **📈 Analytics Insights**: Visualization of sentiment scores (Positive/Negative/Neutral) and domain trust ratings.
3.  **📉 Model Performance**: Detailed metrics (94.2% Accuracy) with interactive Radar charts and Confusion Matrix.
4.  **🛡️ Data / EDA Admin**: High-level training data overview showing class balance and emerging keyword clouds.
5.  **🕒 Recent History**: Persistent history of your last 10 analyzed articles for quick reference.

---

## 🛠️ Tech Stack

### Backend & ML
- **Python / Flask**: Robust backend API and routing.
- **Scikit-learn**: Logistic Regression and TF-IDF pipeline for high-precision classification.
- **NLTK / TextBlob**: Natural Language Processing and Sentiment Analysis.

### Content & Translation
- **Deep Translator**: Seamless bidirectional translation for global language support.
- **Newspaper3k / BS4**: High-fidelity article extraction from news URLs.
- **ReportLab**: Dynamic PDF generation for analysis reports.

### UI & Frontend
- **Vanilla CSS (Glassmorphism)**: Modern, premium interface with backdrop blur and vibrant gradients.
- **Chart.js**: Interactive, responsive data visualizations.
- **Dark/Light Mode**: Full theme customization support.

---

## 💻 How to Use

### Web Interface
1.  **Select Language**: Use the sidebar to switch between English, Hindi, or Telugu.
2.  **Input News**: Paste a news article URL or the full article text in the Analyzer.
3.  **Analyze**: Click **"Analyze Now"**.
4.  **Explore**: Use the sidebar to navigate between analytics, performance, and data distribution.
5.  **Export**: Preview the analysis report and download it as a PDF.

### REST API
**Endpoint:** `POST /predict`
```json
{
  "text": "Paste news article or URL here",
  "is_url": false,
  "target_lang": "te"
}
```

---

## 📂 Project Structure

```text
fake_news_detection_media_integrity/
├── app.py                  # Main Flask App
├── src/
│   ├── inference/          # ML Prediction Logic
│   ├── api/                # Analytics & Batch Processing
│   └── preprocessing/      # Translation & Language Detection
├── artifacts/
│   └── models/             # Pickled ML models & Vectorizers
├── templates/
│   └── index.html          # Single-page Glassmorphism Frontend
├── notebooks/              # Detailed EDA and Training Workbooks
└── requirements.txt        # Project Dependencies
```

---

## ⚙️ Installation & Setup

1.  **Clone the Repo**:
    ```bash
    git clone https://github.com/YourUsername/fake-news-detection-media-integrity.git
    cd fake-news-detection-media-integrity
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Locally**:
    ```bash
    python app.py
    ```
4.  **Access App**: Open `http://127.0.0.1:5000` in your browser.

---

## 🤝 Contributing

Contributions are welcome! Whether it's adding more languages, improving model accuracy, or refining the UI, feel free to fork and submit a PR.