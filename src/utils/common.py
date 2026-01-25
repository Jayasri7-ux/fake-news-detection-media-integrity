import pickle
import os

def save_pickle(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
