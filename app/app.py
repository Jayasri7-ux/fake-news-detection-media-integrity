import sys
import os
from flask import Flask, request, render_template, jsonify
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.predictor import FakeNewsPredictor

app = Flask(__name__)
predictor = FakeNewsPredictor()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()
    text = data.get("text", "")
    result = predictor.predict(text)

    return jsonify({
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "mode": result["mode"],
        "reason": result["reason"]
    })

@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    data = request.get_json()

    prediction = data["prediction"]
    confidence = round(data["confidence"] * 100)
    mode = data["mode"]
    reason = data["reason"]
    chart_base64 = data["chart"]

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>Fake News Detection Report</b>", styles["Title"]))
    content.append(Paragraph(f"<b>Prediction:</b> {prediction}", styles["Normal"]))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence} %", styles["Normal"]))
    content.append(Paragraph(f"<b>Mode:</b> {mode}", styles["Normal"]))
    content.append(Paragraph(f"<b>Reason:</b> {reason}", styles["Normal"]))

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

if __name__ == "__main__":
    app.run(debug=True)
