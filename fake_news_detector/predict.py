"""
predict.py
----------
Loads the trained model and lets you check whether a news headline/article
is FAKE or REAL.

Run interactively:
    python predict.py

Or pass text directly as a command-line argument:
    python predict.py "Scientists confirm shocking cure found in your kitchen"
"""

import sys
import re
import string
import joblib

MODEL_PATH = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        print("Model files not found. Please run 'python train.py' first.")
        sys.exit(1)


def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    confidence = max(proba) * 100
    return prediction, confidence, dict(zip(classes, proba))


def main():
    model, vectorizer = load_model()

    # Mode 1: text passed as command-line argument
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        label, confidence, _ = predict(text, model, vectorizer)
        print(f"\nText: {text}")
        print(f"Prediction: {label}  (confidence: {confidence:.2f}%)\n")
        return

    # Mode 2: interactive loop
    print("=== Fake News Detector ===")
    print("Type a news headline/article and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        text = input("Enter news text: ").strip()
        if text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not text:
            continue
        label, confidence, _ = predict(text, model, vectorizer)
        print(f"--> Prediction: {label}  (confidence: {confidence:.2f}%)\n")


if __name__ == "__main__":
    main()
