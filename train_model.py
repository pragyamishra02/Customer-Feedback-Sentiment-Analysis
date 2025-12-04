"""
train_model.py
Script to train the model from Customer_Feedback_500.xlsx
Usage: python train_model.py
"""
import pandas as pd, joblib, json, re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in ENGLISH_STOP_WORDS and len(t)>1]
    return " ".join(tokens)

df = pd.read_excel("Customer_Feedback_500.xlsx")
df = df.dropna(subset=[df.columns[1], df.columns[2]])
df = df.rename(columns={df.columns[0]:'Customer_ID', df.columns[1]:'Feedback_Text', df.columns[2]:'Sentiment'})
df['text_clean'] = df['Feedback_Text'].apply(preprocess_text)
def norm_label(l):
    l = str(l).strip().lower()
    if l.startswith('pos'): return 'Positive'
    if l.startswith('neg'): return 'Negative'
    if l.startswith('neu'): return 'Neutral'
    if 'good' in l or 'happy' in l or 'satis' in l or 'excellent' in l: return 'Positive'
    if 'bad' in l or 'poor' in l or 'disappoint' in l or 'damag' in l: return 'Negative'
    return 'Neutral'
df['label'] = df['Sentiment'].apply(norm_label)
X = df['text_clean'].values; y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=['Positive','Negative','Neutral'])
report = classification_report(y_test, y_pred, digits=4)
print("Accuracy:", acc)
print("Classification report:\\n", report)
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")
