import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")
MODEL_DIR = os.path.join(project_root, "models")

print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)
print("MODEL DIR:", MODEL_DIR)

print("\n========== MODEL TRAINING COMPARISON ==========")

# --------------------------------------------------
# LOAD TRAIN / TEST DATA
# --------------------------------------------------
X_train = joblib.load(os.path.join(PROCESSED_DATA_PATH, "X_train.pkl"))
X_test = joblib.load(os.path.join(PROCESSED_DATA_PATH, "X_test.pkl"))
y_train = joblib.load(os.path.join(PROCESSED_DATA_PATH, "y_train.pkl"))
y_test = joblib.load(os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"))

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# --------------------------------------------------
# MODELS TO COMPARE
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = []

# --------------------------------------------------
# TRAIN & EVALUATE
# --------------------------------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    results.append((name, acc))

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
print("\n========== COMPARISON SUMMARY ==========")
print(results_df.sort_values(by="Accuracy", ascending=False))

print("\nModel training comparison completed successfully")
