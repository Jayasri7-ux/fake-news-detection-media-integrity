# Fake News Detection for Media Integrity

A Machine Learning–based system to classify news articles as **Fake** or **Real** by analyzing textual content.  
The project follows an end-to-end ML pipeline with data processing, visualization, model evaluation, and a Flask-based web application for deployment.

---

## 📖 Project Overview

The rapid spread of fake news on digital platforms negatively impacts public trust and informed decision-making.  
This project aims to automatically detect fake news articles using Natural Language Processing (NLP) and Machine Learning techniques.

The system performs:
- News data collection and validation
- Text preprocessing and normalization
- Feature extraction using TF-IDF
- Model training, evaluation, and comparison
- Visualization of insights
- Web-based prediction using Flask

---
## 📂 Project Resources
- 🎥 [Demo Video] (https://drive.google.com/file/d/1NF2-Ve5NvLc_uw0y0kTO8vVb2_3cgvO3/view?usp=sharing)
- 🎥 [Demo Video] (https://drive.google.com/file/d/1G2DRAC7bHy46bVpic5-El8uweLD8AR3W/view?usp=sharing)

## 🧰 Technologies Used

- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - Flask
  - wordcloud
  - langdetect
- **Tools:** Git, GitHub

---

## 📁 Project Structure

fake-news-detection-media-integrity/
│
├── app/ # Flask application
│ ├── app.py
│ └── templates/
│ └── index.html
│
├── artifacts/ # Generated models and reports
│ ├── models/
│ └── reports/
│
├── data/ # Dataset storage
│ ├── raw/
│ └── processed/
│
├── notebooks/ # Jupyter notebooks (experimentation)
│ ├── 01_data_collection.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_data_quality_report.ipynb
│ ├── 04_eda.ipynb
│ ├── 05_feature_engineering.ipynb
│ ├── 06_model_training.ipynb
│ ├── 07_model_evaluation.ipynb
│ ├── 08_model_comparison.ipynb
│ ├── 09_model_training_comparison.ipynb
│ └── 10_eda_dashboard.ipynb
│
├── extracted_scripts/ # Python scripts extracted from notebooks
│ ├── 01_data_collection.py
│ ├── 02_data_cleaning.py
│ ├── 03_data_quality_report.py
│ ├── 04_eda.py
│ ├── 05_feature_engineering.py
│ ├── 06_model_training.py
│ ├── 07_model_evaluation.py
│ ├── 08_model_comparison.py
│ ├── 09_model_training_comparison.py
│ └── 10_eda_dashboard.py
│
├── src/ # Core modular ML pipeline
│ ├── data/
│ ├── preprocessing/
│ ├── features/
│ ├── models/
│ ├── inference/
│ ├── utils/
│ └── visuals/
│
├── visuals/
│ ├── plots/
│ └── wordclouds/
│
├── accuracy_plot.py
├── create_wordcloud.py
├── inspect_data.py
├── model_comparison_report.md
├── main.py # Pipeline execution entry point
├── predict.py # Standalone prediction script
├── requirements.txt
├── .gitignore
└── README.md

---

## 📊 Visualizations

- Word Cloud showing frequently occurring words in fake news
- Accuracy comparison plot of different machine learning models
- Exploratory Data Analysis (EDA) visualizations

---

## 🤖 Machine Learning Approach

- Text preprocessing and normalization
- Language detection and translation (if required)
- TF-IDF vectorization
- Logistic Regression model
- Model evaluation using accuracy and comparison metrics

---

## 🌐 Flask Web Application

The Flask-based web application allows users to:
- Enter news text
- Get instant predictions as **Fake** or **Real**

This demonstrates the deployment of the trained ML model in a real-world scenario.

---

## ▶️ How to Run the Project

### Step 1: Clone the repository
```bash
git clone https://github.com/Jayasri7-ux/fake-news-detection-media-integrity.git
cd fake-news-detection-media-integrity
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

Conclusion

This project demonstrates a complete Machine Learning workflow—from data processing and model training to evaluation and deployment—providing an effective solution for detecting fake news and promoting media integrity.
