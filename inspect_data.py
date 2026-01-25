import pandas as pd
import os
import sys

def inspect_file(path):
    print(f"--- Inspecting {os.path.basename(path)} ---")
    if not os.path.exists(path):
        print("File does not exist.")
        return

    try:
        if path.endswith('.pkl'):
            data = pd.read_pickle(path)
        else:
            data = pd.read_csv(path)
            
        if hasattr(data, 'shape'):
            print(f"Shape: {data.shape}")
            
        if hasattr(data, 'columns'):
            print(f"Columns: {list(data.columns)[:10]}")
            print("Dtypes:")
            print(data.dtypes.head(5))
        elif hasattr(data, 'toarray'):
            print("Type: SciPy Sparse Matrix (or similar)")
            print(f"Shape: {data.shape}")
        else:
            print(f"Type: {type(data)}")
            
    except Exception as e:
        print(f"Error reading file: {e}")
    print("\n")

inspect_file(r"c:\project\fake_news_detection_media_integrity\data\processed\cleaned_data.csv")
inspect_file(r"c:\project\fake_news_detection_media_integrity\data\processed\X_train.pkl")
