import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# extracted_scripts -> fake_news_detection_media_integrity
project_root = os.path.abspath(os.path.join(script_dir, ".."))

PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")

print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data_path = os.path.join(PROCESSED_DATA_PATH, "combined_raw_data.csv")
df_raw = pd.read_csv(data_path)

print("\n========== DATA QUALITY REPORT ==========")

# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------
print("\n--- BASIC INFORMATION ---")
print("Shape:", df_raw.shape)
print("\nData Types:")
print(df_raw.dtypes)

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------
print("\n--- MISSING VALUES ---")
missing = df_raw.isnull().sum()
missing_percent = (missing / len(df_raw)) * 100

missing_df = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_percent.round(2)
})

print(missing_df[missing_df["Missing_Count"] > 0])

# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------
print("\n--- DUPLICATE CHECK ---")
duplicate_rows = df_raw.duplicated().sum()
print("Duplicate rows:", duplicate_rows)

# --------------------------------------------------
# LABEL DISTRIBUTION
# --------------------------------------------------
print("\n--- LABEL DISTRIBUTION ---")
print(df_raw["label"].value_counts(dropna=False))

# --------------------------------------------------
# TEXT LENGTH ANALYSIS (SAFE & FUTURE-PROOF)
# --------------------------------------------------
print("\n--- TEXT LENGTH STATS ---")

# Defensive text handling (CRITICAL FIX)
df_raw["text"] = df_raw["text"].fillna("").astype(str)

df_raw["text_length"] = df_raw["text"].apply(len)

print(df_raw["text_length"].describe())

# --------------------------------------------------
# QUALITY SUMMARY
# --------------------------------------------------
print("\n========== QUALITY SUMMARY ==========")
print(f"Total Rows        : {df_raw.shape[0]}")
print(f"Total Columns     : {df_raw.shape[1]}")
print(f"Rows with Nulls   : {(df_raw.isnull().any(axis=1)).sum()}")
print(f"Duplicate Rows    : {duplicate_rows}")

print("\nSample records:")
print(df_raw.head())
