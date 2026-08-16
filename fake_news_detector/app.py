"""
app.py
------
A tiny Flask web app so you can paste a headline into the browser and get
a FAKE / REAL verdict, instead of using the terminal.

Run:
    python app.py

Then open your browser at:
    http://127.0.0.1:5000
"""

import re
import string

import joblib
from flask import Flask, jsonify, render_template, request

MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

app = Flask(__name__)

# Load the trained model + vectorizer once, when the server starts
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    model, vectorizer = None, None


def clean_text(text):
    """Same cleaning logic used during training (train.py)."""
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def home():
    """Renders the main page with the input box."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receives {"text": "..."} and returns a FAKE/REAL verdict as JSON."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not found. Run 'python train.py' first."}), 500

    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    confidence = round(max(proba) * 100, 2)

    return jsonify({"label": label, "confidence": confidence})


if __name__ == "__main__":
    # debug=True auto-reloads the server when you edit the code (dev only)
    app.run(debug=True)
