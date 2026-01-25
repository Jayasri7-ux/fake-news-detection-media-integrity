import pandas as pd
import os

class DataLoader:
    def load_csv(self, path, encoding=None):
        if encoding:
            return pd.read_csv(path, encoding=encoding)
        return pd.read_csv(path)

    def load_excel(self, path):
        return pd.read_excel(path)

    def load_pickle(self, path):
        return pd.read_pickle(path)
