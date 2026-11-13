"""Train, compare and save Logistic Regression and Multinomial Naive Bayes
on TF-IDF features.

Usage:
    python -m src.train                       # uses data/Fake.csv + data/True.csv
    python -m src.train --data-dir data/sample
    python -m src.train --data my.csv         # columns: text[,title], label (1=fake)
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from src.preprocess import clean_text

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Real", "Fake"]  # index = label value (fake = 1 = positive class)


def load_data(data_dir: Path, data_file: str | None) -> pd.DataFrame:
    if data_file:
        df = pd.read_csv(data_file)
    else:
        fake = pd.read_csv(data_dir / "Fake.csv").assign(label=1)
        real = pd.read_csv(data_dir / "True.csv").assign(label=0)
        df = pd.concat([fake, real], ignore_index=True)
    if "title" not in df.columns:
        df["title"] = ""
    df["raw"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["clean"] = df["raw"].map(clean_text)
    df = df[df["clean"].str.len() > 0]
    # Duplicates across train/test would inflate scores.
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)
    return df[["clean", "label"]]


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                           max_features=50_000, min_df=2, max_df=0.9,
                           sublinear_tf=True)


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }, pred, proba


def top_features(vec, lr, nb, k=20):
    names = np.array(vec.get_feature_names_out())
    lr_order = np.argsort(lr.coef_[0])
    nb_diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    nb_order = np.argsort(nb_diff)
    rows = []
    for mname, order in (("logistic_regression", lr_order), ("multinomial_nb", nb_order)):
        for w in names[order[-k:][::-1]]:
            rows.append((mname, "fake", w))
        for w in names[order[:k]]:
            rows.append((mname, "real", w))
    return pd.DataFrame(rows, columns=["model", "indicates", "term"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--data", default=None, help="single CSV (text[,title], label)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=5, help="CV folds on the training set (0 to skip)")
    args = ap.parse_args()

    models_dir, reports_dir = ROOT / "models", ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    print("Loading and cleaning data ...")
    df = load_data(Path(args.data_dir), args.data)
    print(f"  {len(df)} articles | fake share: {df['label'].mean():.1%}")

    # Split BEFORE fitting TF-IDF so the test set never influences the vocabulary/IDF.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=args.test_size,
        stratify=df["label"], random_state=SEED)

    print("Fitting TF-IDF ...")
    vec = make_vectorizer()
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"  vocabulary size: {len(vec.vocabulary_)}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10, random_state=SEED),
        "Multinomial NB": MultinomialNB(alpha=0.1),
    }

    results, preds, probas = [], {}, {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(Xtr, y_train)
        metrics, pred, proba = evaluate(name, model, Xte, y_test)
        if args.cv > 1:
            # Pipeline refits TF-IDF inside each fold -> leak-free CV.
            pipe = make_pipeline(make_vectorizer(), model.__class__(**model.get_params()))
            scores = cross_val_score(pipe, X_train, y_train, cv=args.cv, scoring="f1", n_jobs=-1)
            metrics["cv_f1_mean"], metrics["cv_f1_std"] = scores.mean(), scores.std()
        results.append(metrics)
        preds[name], probas[name] = pred, proba
        print(classification_report(y_test, pred, target_names=LABELS, digits=4))

    comparison = pd.DataFrame(results).set_index("model").round(4)
    print("\n=== Model comparison (held-out test set) ===")
    print(comparison.to_string())
    comparison.to_csv(reports_dir / "model_comparison.csv")

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, models):
        ConfusionMatrixDisplay(confusion_matrix(y_test, preds[name]),
                               display_labels=LABELS).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ROC curves
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, probas[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probas[name]):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Interpretability
    lr, nb = models["Logistic Regression"], models["Multinomial NB"]
    top_features(vec, lr, nb).to_csv(reports_dir / "top_features.csv", index=False)

    # Save vectorizer + both models + the best one (by F1, ties -> ROC AUC)
    best = comparison.sort_values(["f1", "roc_auc"], ascending=False).index[0]
    joblib.dump(vec, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(lr, models_dir / "logistic_regression.joblib")
    joblib.dump(nb, models_dir / "multinomial_nb.joblib")
    joblib.dump(models[best], models_dir / "best_model.joblib")

    summary = {"best_model": best, "n_train": len(X_train), "n_test": len(X_test),
               "vocab_size": len(vec.vocabulary_), "metrics": comparison.reset_index().to_dict("records")}
    (models_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest model: {best}\nSaved artifacts to {models_dir} and reports to {reports_dir}")


if __name__ == "__main__":
    main()
