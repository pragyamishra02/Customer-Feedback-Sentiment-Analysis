"""
data_processing.py
Simple preprocessing utilities for the sentiment project.
"""
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ps = PorterStemmer()

stopwords = set(ENGLISH_STOP_WORDS)

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in stopwords and len(t)>1]
    return " ".join(tokens)
