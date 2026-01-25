import os
import pickle

from src.preprocessing.language_detector import LanguageDetector
from src.preprocessing.translator import Translator
from src.preprocessing.context_expander import ContextExpander


class FakeNewsPredictor:
    def __init__(self):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

        model_path = os.path.join(BASE_DIR, "artifacts", "models", "logistic_model.pkl")
        vectorizer_path = os.path.join(BASE_DIR, "artifacts", "models", "tfidf_vectorizer.pkl")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.lang_detector = LanguageDetector()
        self.translator = Translator()
        self.expander = ContextExpander()

        self.short_real_keywords = [
            "rain", "flood", "earthquake", "power", "train", "bus",
            "government", "school", "college", "exam", "hospital",
            "vaccine", "weather", "traffic", "price", "petrol",
            "diesel", "ration", "salary", "jobs", "scheme", "policy",
            "minister", "cm", "pm", "budget", "education"
        ]

        self.identity_patterns = [
            "i am the cm",
            "i am the pm",
            "i am the president",
            "i control the government",
            "i own the country"
        ]

        self.fake_patterns = [
            "aliens",
            "ufo",
            "everyone will get",
            "free gold",
            "free money",
            "earth swallowed",
            "sun stopped",
            "entire city destroyed",
            "miracle cure",
            "secret formula",
            "100% guaranteed",
            "magic"
        ]

    def rule_based_fake_check(self, text):
        text_lower = text.lower()

        for p in self.identity_patterns:
            if p in text_lower:
                return True, "Identity claim detected"

        for p in self.fake_patterns:
            if p in text_lower:
                return True, "Impossible or sensational claim detected"

        return False, None

    def predict(self, text):
        text_lower = text.lower()
        word_count = len(text.split())

        # 1. Rule-based override
        is_fake, rule_reason = self.rule_based_fake_check(text)
        if is_fake:
            return {
                "input_text": text,
                "expanded_text": text,
                "prediction": "Fake",
                "confidence": 0.99,
                "mode": "Rule-based",
                "reason": rule_reason
            }

        # 2. Short-message plausibility logic
        if word_count <= 5:
            plausible = any(word in text_lower for word in self.short_real_keywords)
            if not plausible:
                return {
                    "input_text": text,
                    "expanded_text": text,
                    "prediction": "Fake",
                    "confidence": 0.85,
                    "mode": "Logic-based",
                    "reason": "Short suspicious message detected"
                }

        # 3. ML pipeline
        lang = self.lang_detector.detect_language(text)
        translated_text = self.translator.translate_to_english(text, lang)
        expanded_text = self.expander.expand(translated_text)

        X = self.vectorizer.transform([expanded_text])

        pred = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        return {
            "input_text": text,
            "expanded_text": expanded_text,
            "prediction": "Real" if pred == 1 else "Fake",
            "confidence": round(float(max(probs)), 4),
            "mode": "ML-based",
            "reason": "Pattern-based ML decision"
        }
