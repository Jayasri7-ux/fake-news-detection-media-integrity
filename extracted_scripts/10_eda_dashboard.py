import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.preprocessing import KBinsDiscretizer

sns.set(style="whitegrid")


class EDADashboard:
    def __init__(self, df):
        self.df = df

        # Fix column names if missing
        if hasattr(self.df, "columns") and (
            len(self.df.columns) == 0 or isinstance(self.df.columns[0], int)
        ):
            self.df.columns = [f"col_{i}" for i in range(self.df.shape[1])]

        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        exclude_cols = ['label', 'target', 'id', 'class']
        self.numeric_cols = [c for c in self.numeric_cols if c.lower() not in exclude_cols]

        self.categorical_cols = self.df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

    def generate_summary(self):
        print("\n--- DATA SUMMARY ---")
        print("\nHead:")
        print(self.df.head())

        print("\nInfo:")
        self.df.info()

        print("\nDescription:")
        print(self.df.describe())

        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def analyze_distributions(self):
        print("\n--- DISTRIBUTIONS ---")
        if not self.numeric_cols:
            print("No numeric columns found.")
            return

        display_cols = self.numeric_cols[:5]
        if len(self.numeric_cols) > 5:
            print(f"Showing first 5 numeric columns out of {len(self.numeric_cols)}")

        for col in display_cols:
            plt.figure(figsize=(10, 4))

            plt.subplot(1, 2, 1)
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Histogram of {col}")

            plt.subplot(1, 2, 2)
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")

            plt.tight_layout()
            plt.show()

            try:
                print(f"{col} → Skew: {self.df[col].skew():.2f}, "
                      f"Kurtosis: {self.df[col].kurt():.2f}")
            except Exception:
                pass

    def analyze_correlations(self, threshold=0.8):
        print("\n--- CORRELATION ANALYSIS ---")

        if len(self.numeric_cols) < 2:
            print("Not enough numeric columns.")
            return

        cols = self.numeric_cols[:20] if len(self.numeric_cols) > 20 else self.numeric_cols
        corr = self.df[cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        plt.title("Correlation Matrix")
        plt.show()

        print(f"\nHigh correlations (> {threshold}):")
        pairs = corr.unstack()
        pairs = pairs[(abs(pairs) > threshold) & (abs(pairs) < 1)]

        printed = set()
        for (c1, c2), val in pairs.items():
            if (c2, c1) not in printed:
                print(f"{c1} ↔ {c2}: {val:.2f}")
                printed.add((c1, c2))

    def create_features(self):
        print("\n--- FEATURE ENGINEERING (EDA ONLY) ---")
        df_new = self.df.copy()

        for col in self.numeric_cols[:2]:
            df_new[f"{col}_squared"] = df_new[col] ** 2
            print(f"Created: {col}_squared")

            try:
                est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                df_new[f"{col}_binned"] = est.fit_transform(
                    df_new[[col]]
                ).flatten()
                print(f"Created: {col}_binned")
            except Exception as e:
                print(f"Skipping binning for {col}: {e}")

        return df_new


# --------------------------------------------------
# PATH HANDLING (PIPELINE SAFE)
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed")

print("\nAttempting to load cleaned_data.csv for EDA...")
cleaned_path = os.path.join(PROCESSED_DATA_PATH, "cleaned_data.csv")

if os.path.exists(cleaned_path):
    df = pd.read_csv(cleaned_path)
    print("Data loaded:", df.shape)

    if "clean_text" in df.columns:
        text_col = "clean_text"
    elif "text" in df.columns:
        text_col = "text"
    else:
        text_col = None

    if text_col:
        df[text_col] = df[text_col].fillna("").astype(str)
        df["word_count"] = df[text_col].apply(lambda x: len(x.split()))
        df["char_count"] = df[text_col].apply(len)
        df["avg_word_length"] = df["char_count"] / (df["word_count"] + 1)
        print("Added text-based numeric features")

else:
    raise FileNotFoundError("cleaned_data.csv not found")

# --------------------------------------------------
# RUN DASHBOARD
# --------------------------------------------------
dashboard = EDADashboard(df)

dashboard.generate_summary()
dashboard.analyze_distributions()
dashboard.analyze_correlations()

df_enhanced = dashboard.create_features()

print("\nFirst 5 rows of enhanced dataframe:")
print(df_enhanced.head())

print("\nEDA DASHBOARD COMPLETED SUCCESSFULLY")
