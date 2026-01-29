import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# extracted_scripts -> fake_news_detection_media_integrity
project_root = os.path.abspath(os.path.join(script_dir, ".."))

PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")
MODEL_DIR = os.path.join(project_root, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)
print("MODEL DIR:", MODEL_DIR)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data_path = os.path.join(PROCESSED_DATA_PATH, "cleaned_data.csv")
df = pd.read_csv(data_path)

print("\n========== FEATURE ENGINEERING ==========")
print("Input data shape:", df.shape)

# --------------------------------------------------
# FEATURES & TARGET (SAFE)
# --------------------------------------------------
# Ensure no NaN values for TF-IDF
X = df["clean_text"].fillna("").astype(str)
y = df["label"]

# Remove empty documents (important)
before_rows = X.shape[0]
valid_mask = X.str.strip() != ""
X = X[valid_mask]
y = y[valid_mask]
after_rows = X.shape[0]

print("Removed empty text rows:", before_rows - after_rows)
print("Final rows for TF-IDF:", after_rows)

# --------------------------------------------------
# TF-IDF VECTORIZATION
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF shape:", X_tfidf.shape)

# --------------------------------------------------
# SAVE FEATURES & VECTORIZER
# --------------------------------------------------
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
joblib.dump(vectorizer, vectorizer_path)

print("Saved TF-IDF vectorizer to:", vectorizer_path)
print("Feature engineering completed successfully")
