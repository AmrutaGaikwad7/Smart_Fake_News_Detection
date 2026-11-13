# Fake News Detection using NLP — Project Report

## 1. Objective
Build and compare machine-learning classifiers that label a news article as *fake* or *real* from its text, and
deploy the best one in an interactive web application.

## 2. Dataset
ISOT Fake News Dataset (Kaggle): ~23k fake and ~21k real English-language articles, with `title`, `text`,
`subject` and `date`. Real articles come from Reuters; fake ones from flagged unreliable sites.
After cleaning and de-duplication, the exact counts are printed by `python -m src.train`.

## 3. Methodology
**Preprocessing** (`src/preprocess.py`): title + body concatenated; lowercased; URLs, HTML, digits and punctuation
removed; the `(Reuters)` source tag removed to prevent label leakage; exact duplicates dropped.

**Features**: `TfidfVectorizer` with English stop-word removal, unigrams + bigrams, `max_features=50,000`,
`min_df=2`, `max_df=0.9`, `sublinear_tf=True`. TF-IDF weights a term by how often it appears in a document
and down-weights terms common across all documents, so distinctive words dominate.

**Models**
- *Logistic Regression* (`C=10`, L2): a linear model over TF-IDF; coefficients are directly interpretable.
- *Multinomial Naive Bayes* (`alpha=0.1`): a generative model that assumes word independence; very fast and a
  strong text baseline.

**Evaluation protocol**: stratified 80/20 train/test split (seed 42); vectorizer fit on the training set only;
5-fold stratified CV on the training set (vectorizer refit in each fold); metrics on the untouched test set:
accuracy, precision, recall, F1, ROC-AUC, confusion matrix. *Fake* is the positive class.

## 4. Results
Run `python -m src.train`, then copy the numbers from `reports/model_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (mean ± std) |
|---|---|---|---|---|---|---|
| Logistic Regression | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |
| Multinomial NB | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ | _fill in_ |

Figures: `reports/confusion_matrices.png`, `reports/roc_curves.png`.

## 5. Discussion  (complete after training)
Points worth addressing with your numbers:
- **Which model wins and by how much?** Logistic Regression typically edges out Naive Bayes on this data because it
  learns discriminative weights and handles correlated n-grams; Naive Bayes is faster to train and often close.
- **Error analysis**: inspect which articles are misclassified (the confusion matrices show whether false
  positives or false negatives dominate) and why.
- **Interpretability**: `reports/top_features.csv` lists the most fake-/real-indicating terms. On ISOT, expect
  stylistic cues (e.g., sensational or opinionated wording, specific outlets/topics) rather than factual signals.
- **Dataset bias**: scores near 99% on ISOT reflect how separable the two sources are, not real-world ability to
  detect misinformation. Test on a different dataset to measure generalization.

## 6. Application
`app.py` (Streamlit) loads the saved vectorizer and models, lets the user pick a model, paste an article and see the
prediction, P(fake), and the words that pushed the decision most (coefficient × TF-IDF for LR; log-likelihood ratio ×
TF-IDF for NB). A sidebar shows test metrics and an expander shows evaluation plots.

## 7. Limitations and ethics
- Learns dataset-specific style and topics; may fail on other publishers, languages, satire or breaking news.
- Cannot verify facts or use external evidence; a "Real" label does not mean true.
- Misclassifying legitimate journalism as fake has real costs — use predictions as a signal for human review.

## 8. Future work
Cross-dataset evaluation; hyper-parameter search (`GridSearchCV`); linear SVM / transformer baselines
(e.g., DistilBERT); using metadata and source credibility; calibrated probabilities; Docker + cloud deployment.

## 9. Reproducibility
`pip install -r requirements.txt` → `python -m src.train` → `streamlit run app.py`. Seed 42 throughout.
