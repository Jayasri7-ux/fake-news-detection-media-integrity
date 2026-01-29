import os
import sys
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.inference.predictor import FakeNewsPredictor

# --------------------------------------------------
# NLTK SETUP (SAFE)
# --------------------------------------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# --------------------------------------------------
# TEXT CLEANING (SAME AS TRAINING)
# --------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --------------------------------------------------
# RESULT DISPLAY
# --------------------------------------------------
def print_result(result):
    print("\n" + "=" * 55)
    print("            FAKE NEWS PREDICTION RESULT")
    print("=" * 55)

    pred = result.get("prediction", "unknown")
    conf = result.get("confidence", 0) * 100
    risk = result.get("risk_level", "N/A")
    mode = result.get("mode", "text")

    print(f"Prediction     : [{pred.upper()}]")
    print(f"Confidence     : {conf:.2f}%")
    print(f"Risk Level     : [{risk}]")
    print(f"Decision Mode  : {mode}")
    print("-" * 55)

    if result.get("reason"):
        print(f"Reason         : {result['reason']}")

    if result.get("explanation"):
        print("\nExplanation:")
        for line in result["explanation"]:
            print(f"- {line}")

    print("=" * 55 + "\n")

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
def main():
    print("\n" + "*" * 65)
    print("     Fake News Detection System - Terminal Mode")
    print("*" * 65)

    try:
        print("\nLoading trained model and vectorizer...")
        predictor = FakeNewsPredictor()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not load predictor: {e}")
        return

    while True:
        print("\nEnter news text to check (or type 'exit' to quit):")
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nExiting prediction system. Stay informed!\n")
            break

        try:
            print("\nCleaning input text...")
            cleaned_input = clean_text(user_input)

            if not cleaned_input.strip():
                print("[WARNING] Text became empty after cleaning. Try a longer input.")
                continue

            print("Analyzing text...")
            result = predictor.predict(cleaned_input)
            print_result(result)

        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)

    main()
