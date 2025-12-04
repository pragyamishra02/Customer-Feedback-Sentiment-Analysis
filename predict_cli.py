"""
predict_cli.py
Simple command-line predictor. Usage:
python predict_cli.py "The product arrived late and damaged."
"""
import sys, joblib, re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ps = PorterStemmer()
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("model.joblib")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS and len(t)>1]
    return " ".join(tokens)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Provide text in quotes: python predict_cli.py \"The product was great.\"")
        sys.exit(0)
    text = " ".join(sys.argv[1:])
    cleaned = preprocess_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    print("Predicted sentiment:", pred)
