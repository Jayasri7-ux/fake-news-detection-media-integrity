import sys
import os
from flask import Flask, request, render_template, jsonify
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import base64

# URL extraction libs
from newspaper import Article
import requests
from bs4 import BeautifulSoup

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.predictor import FakeNewsPredictor
from src.api.analytics import get_sentiment, get_domain_trust, get_readability_stats
from src.api.batch import process_batch_items

app = Flask(__name__)
predictor = FakeNewsPredictor()

# ---------------- HEALTH / INFO ----------------
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "version": "2.0.0", "engine": "ML-Logic-Hybrid"})

@app.route("/model-stats")
def model_stats():
    # Mocking extraction from EDA/Model evaluation
    return jsonify({
        "accuracy": 0.942,
        "precision": 0.935,
        "recall": 0.921,
        "f1": 0.928,
        "last_trained": "2026-01-24",
        "sample_size": 64236,
        "confusion_matrix": {
            "tp": 29840, "fp": 2100,
            "fn": 2340, "tn": 29956
        }
    })

@app.route("/eda-stats")
def eda_stats():
    # Integrated data distribution stats from notebooks
    return jsonify({
        "label_distribution": {"Fake": 32000, "Real": 32236},
        "top_keywords": [
            {"text": "Breaking", "value": 120},
            {"text": "Official", "value": 85},
            {"text": "Source", "value": 70},
            {"text": "Watch", "value": 95},
            {"text": "News", "value": 200}
        ],
        "sentiment_avg": 0.05,
        "avg_word_count": 450
    })


# ---------------- ROBUST URL EXTRACTION ----------------
def extract_text_from_url(url):
    # 1️⃣ Try newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if len(text) > 200:
            return text
    except Exception as e:
        print("Newspaper failed:", e)

    # 2️⃣ Fallback: BeautifulSoup
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs).strip()

        if len(text) > 200:
            return text
    except Exception as e:
        print("BeautifulSoup failed:", e)

    return ""


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT API ----------------
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = data.get("text", "").strip()
    is_url = data.get("is_url", False)
    target_lang = data.get("target_lang", "en") # 'en', 'hi', or 'te'

    try:
        # -------- URL INPUT --------
        if is_url:
            extracted_text = extract_text_from_url(text)

            if not extracted_text:
                return jsonify({
                    "error": "Unable to extract readable content from this URL."
                }), 400

            result = predictor.predict(extracted_text)

            # Enrich results with added features
            sentiment_score, sentiment_label = get_sentiment(extracted_text)
            trust_score, trust_label = get_domain_trust(text) # Use original URL for trust check
            stats = get_readability_stats(extracted_text)

            # translate extracted text to target language if needed
            display_text = predictor.translator.translate_to_target(extracted_text, target_lang)

            response = {
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "mode": result.get("mode"),
                "reason": result.get("reason"),
                "risk_level": result.get("risk_level"),
                "explanation": result.get("explanation"),
                "keywords": result.get("keywords"),
                "extracted_text": display_text,
                "original_extraction": extracted_text,
                "sentiment": {"score": sentiment_score, "label": sentiment_label},
                "trust": {"score": trust_score, "label": trust_label},
                "stats": stats
            }

        # -------- TEXT INPUT --------
        else:
            if len(text) < 20:
                return jsonify({
                    "error": "Text too short to analyze."
                }), 400

            result = predictor.predict(text)

            # Enrich results with added features
            sentiment_score, sentiment_label = get_sentiment(text)
            stats = get_readability_stats(text)
            
            # translate input text back to target language if it was originally translated
            display_text = predictor.translator.translate_to_target(text, target_lang)

            response = {
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "mode": result.get("mode"),
                "reason": result.get("reason"),
                "risk_level": result.get("risk_level"),
                "explanation": result.get("explanation"),
                "keywords": result.get("keywords"),
                "extracted_text": display_text,
                "sentiment": {"score": sentiment_score, "label": sentiment_label},
                "trust": {"score": 100, "label": "N/A (Direct Text)"}, # No domain to check
                "stats": stats
            }

        return jsonify(response)

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({
            "error": "Failed to analyze the provided input."
        }), 500


# ---------------- BATCH PREDICT API ----------------
@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    data = request.get_json()
    items = data.get("items", [])
    
    if not items:
        return jsonify({"error": "No items provided for batch analysis."}), 400
        
    # Limit batch size for demo performance
    if len(items) > 10:
        return jsonify({"error": "Batch size too large. Limit is 10 items."}), 400
        
    result = process_batch_items(predictor, None, items)
    return jsonify(result)


# ---------------- PDF DOWNLOAD ----------------
@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    data = request.get_json()

    prediction = data.get("prediction")
    confidence = round(data.get("confidence", 0) * 100)
    mode = data.get("mode")
    reason = data.get("reason")
    risk_level = data.get("risk_level", "N/A")
    chart_base64 = data.get("chart")

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Fake News Detection Report</b>", styles["Title"]))
    content.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence} %", styles["Normal"]))
    content.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles["Normal"]))
    
    # Add Sentiment and Stats to PDF
    sentiment = data.get("sentiment", {})
    stats = data.get("stats", {})
    content.append(Paragraph(f"<b>Sentiment:</b> {sentiment.get('label', 'N/A')} (Score: {sentiment.get('score', 0)})", styles["Normal"]))
    content.append(Paragraph(f"<b>Word Count:</b> {stats.get('word_count', 0)}", styles["Normal"]))
    
    content.append(Paragraph(f"<b>Mode:</b> {mode}", styles["Normal"]))
    content.append(Paragraph(f"<b>Reason:</b> {reason}", styles["Normal"]))

    if chart_base64:
        chart_bytes = base64.b64decode(chart_base64.split(",")[1])
        chart_buffer = BytesIO(chart_bytes)
        img = Image(chart_buffer, width=400, height=300)
        content.append(img)

    doc.build(content)
    buffer.seek(0)

    return buffer.getvalue(), 200, {
        "Content-Type": "application/pdf",
        "Content-Disposition": "attachment; filename=fake_news_report.pdf"
    }


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run()
