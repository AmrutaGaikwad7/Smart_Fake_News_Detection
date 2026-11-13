# 📰 Fake News Detection with NLP

Classifies news articles as **Fake** or **Real** using **TF-IDF** features and two classic models —
**Logistic Regression** and **Multinomial Naive Bayes** — compared head-to-head and served through a **Streamlit** app.

## Project structure

```
fake-news-detector/
├── app.py                  # Streamlit web app
├── src/
│   ├── preprocess.py       # text cleaning (shared by training + app)
│   └── train.py            # TF-IDF, training, comparison, plots, saving models
├── scripts/make_sample_data.py   # tiny synthetic data for smoke tests only
├── tests/test_pipeline.py
├── data/                   # put Fake.csv and True.csv here (see data/README.md)
├── models/                 # saved vectorizer + models (created by training)
├── reports/                # metrics CSV, confusion matrices, ROC, top features
├── REPORT.md               # project report
└── requirements.txt
```

## Quick start

```bash
git clone <your-repo-url> && cd fake-news-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Download Fake.csv and True.csv into data/ (see data/README.md)
# 2. Train, compare and save models
python -m src.train
# 3. Launch the app
streamlit run app.py
```

No dataset yet? Smoke-test the pipeline with synthetic data (scores are meaningless):
`python scripts/make_sample_data.py && python -m src.train --data-dir data/sample`

## What training produces

| File | Description |
|---|---|
| `models/tfidf_vectorizer.joblib` | fitted TF-IDF vectorizer |
| `models/logistic_regression.joblib`, `models/multinomial_nb.joblib` | both trained models |
| `models/best_model.joblib` | winner by F1 (ties → ROC-AUC) |
| `models/metrics.json`, `reports/model_comparison.csv` | evaluation numbers |
| `reports/confusion_matrices.png`, `reports/roc_curves.png` | plots |
| `reports/top_features.csv` | most fake-/real-indicating terms per model |

Load them yourself:

```python
import joblib
from src.preprocess import clean_text
vec = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/best_model.joblib")
p_fake = model.predict_proba(vec.transform([clean_text("some article")]))[0, 1]
```

## Design choices that keep results honest

- **Split before vectorizing**: TF-IDF is fit on the training set only; CV uses a pipeline so each fold refits it.
- **Source-leak removal**: the `(Reuters)` dateline is stripped. Otherwise models can score ~99% just by spotting it.
- **Duplicate removal** so identical articles can't appear in both train and test.
- **Stratified split** with fixed seed (`42`) for reproducibility.

## Limitations

The model learns the *writing style and topics* of its training dataset, not factual truth. Expect lower accuracy on
other sources, topics or time periods. Use it as a learning project or a signal, not a fact-checker.

## Publishing to GitHub

```bash
git init && git add . && git commit -m "Initial commit: fake news detector"
git branch -M main
git remote add origin https://github.com/<you>/fake-news-detector.git
git push -u origin main
```
Trained `.joblib` files and CSVs are git-ignored by default; remove those lines from `.gitignore` (or use Git LFS)
if you want to ship the model, e.g. to deploy on Streamlit Community Cloud.

## License
MIT
