
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

EXAMPLES = {
    "— choose an example —": "",
    "Sensational claim": "BREAKING: Shocking secret exposed! Insiders reveal the miracle cover-up the "
                         "mainstream media doesn't want you to see. Share before they delete this!",
    "Wire-style report": "The finance ministry said on Tuesday that lawmakers approved the quarterly budget "
                         "after weeks of negotiations, according to a statement from a committee spokesman.",
}


@st.cache_resource
def load_artifacts():
    vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    models = {
        "Logistic Regression": joblib.load(MODELS / "logistic_regression.joblib"),
        "Multinomial Naive Bayes": joblib.load(MODELS / "multinomial_nb.joblib"),
    }
    return vec, models


def explain(model_name, model, vec, x, k=8):
    """Terms in this article that push the prediction most toward Fake / Real."""
    names = np.array(vec.get_feature_names_out())
    idx = x.nonzero()[1]
    if len(idx) == 0:
        return pd.DataFrame(columns=["term", "pushes toward", "score"])
    if model_name == "Logistic Regression":
        score = model.coef_[0][idx] * x[0, idx].toarray().ravel()
    else:  # Naive Bayes: log-likelihood ratio fake vs real, weighted by TF-IDF
        score = (model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]) * x[0, idx].toarray().ravel()
    df = pd.DataFrame({"term": names[idx], "score": score})
    df["pushes toward"] = np.where(df["score"] > 0, "Fake", "Real")
    df = df.reindex(df["score"].abs().sort_values(ascending=False).index).head(k)
    return df[["term", "pushes toward", "score"]].round(4).reset_index(drop=True)


st.title("📰 Fake News Detector")
st.caption("TF-IDF features + Logistic Regression / Multinomial Naive Bayes")

if not (MODELS / "tfidf_vectorizer.joblib").exists():
    st.error("No trained model found. Run `python -m src.train` first (see README).")
    st.stop()

vec, models = load_artifacts()

with st.sidebar:
    st.header("Settings")
    model_name = st.radio("Model", list(models))
    st.divider()
    metrics_file = MODELS / "metrics.json"
    if metrics_file.exists():
        info = json.loads(metrics_file.read_text())
        st.subheader("Test-set results")
        st.dataframe(pd.DataFrame(info["metrics"]).set_index("model")
                     [["accuracy", "precision", "recall", "f1", "roc_auc"]], width="stretch")
        st.caption(f"Best model: **{info['best_model']}**")

choice = st.selectbox("Try an example", list(EXAMPLES))
text = st.text_area("Paste a news headline and/or article body", value=EXAMPLES[choice],
                    height=220, placeholder="Paste article text here...")

if st.button("Analyze", type="primary"):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 5:
        st.warning("Please enter at least a few words of text.")
    else:
        x = vec.transform([cleaned])
        model = models[model_name]
        if x.nnz == 0:
            st.warning("None of these words are in the model's vocabulary; the result would be meaningless.")
            st.stop()
        p_fake = float(model.predict_proba(x)[0, 1])
        label = "FAKE" if p_fake >= 0.5 else "REAL"
        conf = max(p_fake, 1 - p_fake)
        (st.error if label == "FAKE" else st.success)(f"**Prediction: {label}**  ·  confidence {conf:.1%}")
        st.progress(p_fake, text=f"P(fake) = {p_fake:.1%}")
        st.subheader("Most influential words")
        st.dataframe(explain(model_name, model, vec, x), width="stretch", hide_index=True)
        st.caption("⚠️ The model learned the *style* of one dataset. It cannot verify facts and can be wrong, "
                   "especially on topics or writing styles unlike its training data.")

if (REPORTS / "confusion_matrices.png").exists():
    with st.expander("Model evaluation plots"):
        st.image(str(REPORTS / "confusion_matrices.png"))
        if (REPORTS / "roc_curves.png").exists():
            st.image(str(REPORTS / "roc_curves.png"))
