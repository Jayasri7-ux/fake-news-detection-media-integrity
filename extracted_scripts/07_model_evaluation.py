import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")
MODEL_DIR = os.path.join(project_root, "models")

print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)
print("MODEL DIR:", MODEL_DIR)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
model = joblib.load(model_path)

print("\n========== MODEL EVALUATION ==========")

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------
X_test = joblib.load(os.path.join(PROCESSED_DATA_PATH, "X_test.pkl"))
y_test = joblib.load(os.path.join(PROCESSED_DATA_PATH, "y_test.pkl"))

print("Test data shape:", X_test.shape)

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nModel evaluation completed successfully")
