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

        # ---------------- EXISTING KEYWORDS (UNCHANGED) ----------------
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

    # ---------------- EXISTING RULE CHECK (UNCHANGED) ----------------
    def rule_based_fake_check(self, text):
        text_lower = text.lower()

        for p in self.identity_patterns:
            if p in text_lower:
                return True, "Identity claim detected"

        for p in self.fake_patterns:
            if p in text_lower:
                return True, "Impossible or sensational claim detected"

        return False, None

    # ---------------- NEW SAFE HELPERS ----------------
    def _calculate_risk_level(self, prediction, confidence, mode):
        """
        Centralized risk logic (used everywhere)
        """
        if prediction == "Fake":
            if mode == "Rule-based":
                return "High"
            if confidence >= 0.80:
                return "High"
            elif confidence >= 0.50:
                return "Medium"
            else:
                return "Low"
        return "Low"

    def _extract_keywords(self, text):
        text_lower = text.lower()

        return {
            "fake": [w for w in self.fake_patterns if w in text_lower],
            "real": [w for w in self.short_real_keywords if w in text_lower]
        }

    # ---------------- MAIN PREDICT METHOD ----------------
    def predict(self, text):
        text_lower = text.lower()
        word_count = len(text.split())

        explanation = []
        keywords = self._extract_keywords(text)

        # 1️⃣ RULE-BASED OVERRIDE (UNCHANGED BEHAVIOR)
        is_fake, rule_reason = self.rule_based_fake_check(text)
        if is_fake:
            explanation.extend([
                "Rule-based validation triggered",
                rule_reason
            ])

            return {
                "prediction": "Fake",
                "confidence": 0.99,
                "mode": "Rule-based",
                "reason": rule_reason,
                "risk_level": "High",
                "explanation": explanation,
                "keywords": keywords
            }

        explanation.append("Rule-based validation passed")

        # 2️⃣ SHORT MESSAGE LOGIC (UNCHANGED BEHAVIOR)
        if word_count <= 5:
            plausible = any(word in text_lower for word in self.short_real_keywords)
            if not plausible:
                explanation.append("Short suspicious message detected")

                return {
                    "prediction": "Fake",
                    "confidence": 0.85,
                    "mode": "Logic-based",
                    "reason": "Short suspicious message detected",
                    "risk_level": "Medium",
                    "explanation": explanation,
                    "keywords": keywords
                }

            explanation.append("Short message plausibility check passed")

        # 3️⃣ ML PIPELINE (UNCHANGED CORE ML)
        lang = self.lang_detector.detect_language(text)
        explanation.append(f"Detected language: {lang}")

        translated_text = self.translator.translate_to_english(text, lang)
        explanation.append("Text translated to English")

        expanded_text = self.expander.expand(translated_text)
        explanation.append("Context expansion applied")

        X = self.vectorizer.transform([expanded_text])

        pred = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]

        confidence = round(float(max(probs)), 4)
        prediction = "Real" if pred == 1 else "Fake"

        explanation.extend([
            "TF-IDF vectorization applied",
            "Logistic Regression model applied"
        ])

        risk_level = self._calculate_risk_level(prediction, confidence, "ML-based")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "mode": "ML-based",
            "reason": "Pattern-based ML decision",
            "risk_level": risk_level,
            "explanation": explanation,
            "keywords": keywords
        }
