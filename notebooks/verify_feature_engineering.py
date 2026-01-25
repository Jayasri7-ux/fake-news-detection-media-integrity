import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Mock display for script environment
def display(obj):
    print(obj)

print("Starting verification...")

# Load Data
try:
    df = pd.read_csv("../data/processed/cleaned_data.csv")
    print("Before Feature Engineering - Shape:", df.shape)
    print(df.head())
except FileNotFoundError:
    print("Error: ../data/processed/cleaned_data.csv not found.")
    exit(1)

# Clean Data
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df = df[df["clean_text"].str.strip() != ""]

# X/y
X_text = df["clean_text"]
y = df["label"]

# Feature Engineering
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(X_text)
print("After Feature Engineering - Shape:", X_tfidf.shape)

# Sample of engineered features
feature_names = tfidf.get_feature_names_out()
tfidf_df_sample = pd.DataFrame(X_tfidf[:5].toarray(), columns=feature_names)
print("Sample of engineered features (first 5 rows):")
display(tfidf_df_sample.head())

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# Save Matrices
os.makedirs("../data/processed", exist_ok=True)
with open("../data/processed/X_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
with open("../data/processed/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
with open("../data/processed/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)
with open("../data/processed/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
print("Feature matrices saved.")

# Save Vectorizer
os.makedirs("../artifacts/models", exist_ok=True)
with open("../artifacts/models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("TF-IDF vectorizer saved.")

# Plot and Save
avg_scores = X_tfidf.mean(axis=0).A1
top_indices = avg_scores.argsort()[::-1][:20]
top_features = feature_names[top_indices]
top_scores = avg_scores[top_indices]

plt.figure(figsize=(10, 8))
plt.barh(top_features, top_scores, color='skyblue')
plt.xlabel("Average TF-IDF Score")
plt.title("Top 20 Engineered Features by Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../artifacts/feature_importance.png")
print(f"Feature importance plot saved to {os.path.abspath('../artifacts/feature_importance.png')}")

# Save Insights
insights_text = f"""
# Feature Engineering Insights

## Dataset Dimensions
- **Original Shape**: {df.shape}
- **Engineered Feature Matrix Shape**: {X_tfidf.shape}

## Feature Statistics
- **Total Features**: {len(feature_names)}
- **Top 5 Features by Importance**: {', '.join(top_features[:5])}
- **Sparsity**: {100 * (1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])):.2f}%
"""

with open("../artifacts/feature_engineering_insights.md", "w") as f:
    f.write(insights_text)
print(f"Insights saved to {os.path.abspath('../artifacts/feature_engineering_insights.md')}")

print("Verification complete.")
