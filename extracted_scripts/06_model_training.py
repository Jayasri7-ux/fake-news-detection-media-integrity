import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
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

print("\n========== MODEL TRAINING ==========")
print("Input data shape:", df.shape)

# --------------------------------------------------
# LOAD TF-IDF VECTORIZER
# --------------------------------------------------
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# --------------------------------------------------
# FEATURES & TARGET
# --------------------------------------------------
X = df["clean_text"].fillna("").astype(str)
y = df["label"]

# Remove empty texts
mask = X.str.strip() != ""
X = X[mask]
y = y[mask]

# Vectorize
X_tfidf = vectorizer.transform(X)

print("TF-IDF shape:", X_tfidf.shape)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])

# --------------------------------------------------
# SAVE SPLITS (FOR NEXT STEPS)
# --------------------------------------------------
joblib.dump(X_train, os.path.join(PROCESSED_DATA_PATH, "X_train.pkl"))
joblib.dump(X_test, os.path.join(PROCESSED_DATA_PATH, "X_test.pkl"))
joblib.dump(y_train, os.path.join(PROCESSED_DATA_PATH, "y_train.pkl"))
joblib.dump(y_test, os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"))

print("Saved train-test splits")

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------------
# QUICK TRAIN ACCURACY
# --------------------------------------------------
train_preds = model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)

print("Training Accuracy:", train_acc)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
joblib.dump(model, model_path)

print("Saved trained model to:", model_path)
print("Model training completed successfully")
