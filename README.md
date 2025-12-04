# Customer Feedback Sentiment Analysis - Simple Project

This project trains a simple TF-IDF + Logistic Regression classifier to predict sentiment
(Positive / Negative / Neutral) from customer feedback text.

Dataset used: Customer_Feedback_500.xlsx. (Provided by user in the assignment). fileciteturn0file0

## Files included
- model.joblib           : Trained model
- vectorizer.joblib      : TF-IDF vectorizer
- metrics.json           : Accuracy, confusion matrix and classification report
- assets/                : Confusion matrix and top-word charts
- data_processing.py     : Text preprocessing utilities
- train_model.py         : Script to retrain model from the Excel file
- predict_cli.py         : Simple CLI prediction script
- requirements.txt       : Python dependencies

## How to run in VS Code (Windows / Linux / Mac)
1. Open VS Code and open this project folder.
2. Create a Python virtual environment:
   - Windows (PowerShell): `python -m venv .venv; .\.venv\Scripts\Activate.ps1`
   - Windows (cmd): `python -m venv .venv && .venv\Scripts\activate`
   - macOS / Linux: `python3 -m venv .venv && source .venv/bin/activate`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. (Optional) If running for the first time, you may need to download NLTK punkt or other data, but this project only uses PorterStemmer which does not require downloads.
5. To retrain the model from the provided Excel file: `python train_model.py`
6. To predict a single sentence: `python predict_cli.py "The delivery was quick and staff were polite."`
7. View evaluation metrics in `metrics.json` and images in `assets/`.

## Notes
- Preprocessing uses a simple Porter stemmer and sklearn's built-in English stopwords.
- Word-cloud was replaced by top-word horizontal bar charts to avoid extra dependencies; charts are in `assets/`.
- If you want a GUI or web app, let me know and I can add a minimal Streamlit or Flask app.
