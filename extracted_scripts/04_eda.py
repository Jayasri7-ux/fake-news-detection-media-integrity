import pandas as pd
import os

# --------------------------------------------------
# PATH HANDLING
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")

print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data_path = os.path.join(PROCESSED_DATA_PATH, "cleaned_data.csv")
df = pd.read_csv(data_path)

print("\n========== EDA REPORT ==========")

# --------------------------------------------------
# BASIC INFO
# --------------------------------------------------
print("\n--- BASIC INFO ---")
print("Shape:", df.shape)
print(df.info())

# --------------------------------------------------
# LABEL DISTRIBUTION
# --------------------------------------------------
print("\n--- LABEL DISTRIBUTION ---")
print(df["label"].value_counts())

# --------------------------------------------------
# TEXT LENGTH ANALYSIS (CRITICAL FIX)
# --------------------------------------------------
print("\n--- TEXT LENGTH ANALYSIS ---")

# Defensive normalization (THIS PREVENTS ALL FUTURE ERRORS)
df["clean_text"] = df["clean_text"].fillna("").astype(str)

df["text_length"] = df["clean_text"].apply(len)

print(df["text_length"].describe())

# --------------------------------------------------
# SOURCE DISTRIBUTION
# --------------------------------------------------
print("\n--- SOURCE DISTRIBUTION ---")
print(df["source"].value_counts())

# --------------------------------------------------
# SAMPLE RECORDS
# --------------------------------------------------
print("\nSample records:")
print(df.head())
