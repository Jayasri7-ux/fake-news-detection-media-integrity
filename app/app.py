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

app = Flask(__name__)
predictor = FakeNewsPredictor()


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

    try:
        # -------- URL INPUT --------
        if is_url:
            extracted_text = extract_text_from_url(text)

            if not extracted_text:
                return jsonify({
                    "error": "Unable to extract readable content from this URL."
                }), 400

            result = predictor.predict(extracted_text)

            # ✅ IMPORTANT: send extracted_text back
            response = {
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "mode": result.get("mode"),
                "reason": result.get("reason"),
                "risk_level": result.get("risk_level"),
                "explanation": result.get("explanation"),
                "keywords": result.get("keywords"),
                "extracted_text": extracted_text
            }

        # -------- TEXT INPUT --------
        else:
            if len(text) < 20:
                return jsonify({
                    "error": "Text too short to analyze."
                }), 400

            result = predictor.predict(text)

            response = {
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
                "mode": result.get("mode"),
                "reason": result.get("reason"),
                "risk_level": result.get("risk_level"),
                "explanation": result.get("explanation"),
                "keywords": result.get("keywords")
            }

        return jsonify(response)

    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({
            "error": "Failed to analyze the provided input."
        }), 500


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
    content.append(Paragraph(f"<b>Mode:</b> {mode}", styles["Normal"]))
    content.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles["Normal"]))
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
    app.run(debug=True)
