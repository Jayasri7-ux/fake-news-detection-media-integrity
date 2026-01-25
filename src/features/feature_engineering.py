from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureEngineer:
    def __init__(self, max_features=5000, ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)
