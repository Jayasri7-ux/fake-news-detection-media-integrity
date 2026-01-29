import pandas as pd
import numpy as np
import re
import string
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))

PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")
print("PROCESSED DATA PATH:", PROCESSED_DATA_PATH)

# --------------------------------------------------
# NLTK SETUP
# --------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
input_path = os.path.join(PROCESSED_DATA_PATH, "combined_raw_data.csv")
df = pd.read_csv(input_path)

print("\n========== DATA CLEANING REPORT ==========")

# BEFORE CLEANING
print("\n--- BEFORE CLEANING ---")
print("Total rows:", df.shape[0])
print("Total columns:", df.shape[1])
print("Missing values:\n", df.isnull().sum())

# --------------------------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --------------------------------------------------
# APPLY CLEANING
# --------------------------------------------------
df["clean_text"] = df["text"].apply(clean_text)

# Remove duplicates
rows_before_duplicates = df.shape[0]
df = df.drop_duplicates(subset="clean_text")
rows_after_duplicates = df.shape[0]

# Remove nulls
rows_before_nulls = df.shape[0]
df = df.dropna(subset=["clean_text", "label"])
rows_after_nulls = df.shape[0]

# --------------------------------------------------
# AFTER CLEANING REPORT
# --------------------------------------------------
print("\n--- CLEANING SUMMARY ---")
print("Duplicate rows removed:", rows_before_duplicates - rows_after_duplicates)
print("Rows removed due to nulls:", rows_before_nulls - rows_after_nulls)

print("\n--- AFTER CLEANING ---")
print("Final rows:", df.shape[0])
print("Final missing values:\n", df.isnull().sum())

cleaned_percentage = ((rows_before_duplicates - df.shape[0]) / rows_before_duplicates) * 100
print(f"Total rows cleaned: {rows_before_duplicates - df.shape[0]} ({cleaned_percentage:.2f}%)")

# --------------------------------------------------
# SAVE CLEANED DATA
# --------------------------------------------------
cleaned_path = os.path.join(PROCESSED_DATA_PATH, "cleaned_data.csv")
df.to_csv(cleaned_path, index=False)

print("\nSaved cleaned data to:", cleaned_path)

# --------------------------------------------------
# LABEL DISTRIBUTION
# --------------------------------------------------
print("\nLabel distribution after cleaning:")
print(df["label"].value_counts())

print("\nSample cleaned rows:")
print(df.head())
