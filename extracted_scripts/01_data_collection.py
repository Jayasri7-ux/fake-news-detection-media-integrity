import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_colwidth", None)

# --------------------------------------------------
# ✅ FINAL CORRECT PATH HANDLING
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# extracted_scripts -> fake_news_detection_media_integrity
project_root = os.path.abspath(os.path.join(script_dir, ".."))

RAW_DATA_PATH = os.path.join(project_root, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

print("RAW DATA PATH:", RAW_DATA_PATH)
print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df_ifnd = pd.read_csv(os.path.join(RAW_DATA_PATH, "IFND.csv"), encoding="latin1")
df_bharat = pd.read_csv(os.path.join(RAW_DATA_PATH, "bharatfakenewskosh_raw.csv"))
df_news = pd.read_csv(os.path.join(RAW_DATA_PATH, "news_dataset.csv"))

print("IFND:", df_ifnd.shape)
print("Bharat:", df_bharat.shape)
print("News:", df_news.shape)

print("IFND Columns:", df_ifnd.columns.tolist())
print("Bharat Columns:", df_bharat.columns.tolist())
print("News Columns:", df_news.columns.tolist())

# --------------------------------------------------
# CLEANING & STANDARDIZATION
# --------------------------------------------------

# IFND
df_ifnd_clean = df_ifnd[["Statement", "Label"]].copy()
df_ifnd_clean.columns = ["text", "label"]
df_ifnd_clean["source"] = "IFND"
df_ifnd_clean["label"] = df_ifnd_clean["label"].map({"TRUE": 1, "FALSE": 0})

# Bharat Fake News Kosh
df_bharat_clean = df_bharat[["Eng_Trans_News_Body", "Label"]].copy()
df_bharat_clean.columns = ["text", "label"]
df_bharat_clean["source"] = "BharatFakeNewsKosh"
df_bharat_clean["label"] = df_bharat_clean["label"].map({True: 1, False: 0})

# News dataset
df_news_clean = df_news.copy()
df_news_clean["source"] = "NewsDataset"
df_news_clean["label"] = df_news_clean["label"].map({"REAL": 1, "FAKE": 0})

# --------------------------------------------------
# COMBINE DATASETS
# --------------------------------------------------
df_combined = pd.concat(
    [df_ifnd_clean, df_bharat_clean, df_news_clean],
    axis=0,
    ignore_index=True
)

print("Combined Shape:", df_combined.shape)
print(df_combined.info())
print(df_combined.isnull().sum())

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
sns.countplot(x="label", data=df_combined)
plt.title("Class Distribution (0 = Fake, 1 = Real)")
plt.show()

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
combined_path = os.path.join(PROCESSED_DATA_PATH, "combined_raw_data.csv")
df_combined.to_csv(combined_path, index=False)

print("Saved combined dataset to:", combined_path)
print("Total Samples:", df_combined.shape[0])
print(df_combined["source"].value_counts())
