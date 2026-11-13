
"""Streamlit Fake News Detector with model evaluation and EDA dashboard."""

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
EDA = ROOT / "outputs" / "visualizations"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

EXAMPLES = {
    "Choose an example": "",
    "Sensational claim": (
        "BREAKING: Shocking secret exposed! Insiders reveal the miracle "
        "cover-up the mainstream media doesn't want you to see. "
        "Share before they delete this!"
    ),
    "Wire-style report": (
        "The finance ministry said on Tuesday that lawmakers approved "
        "the quarterly budget after weeks of negotiations, according "
        "to a statement from a committee spokesman."
    ),
}


@st.cache_resource
def load_artifacts():
    vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    models = {
        "Logistic Regression": joblib.load(
            MODELS / "logistic_regression.joblib"
        ),
        "Multinomial Naive Bayes": joblib.load(
            MODELS / "multinomial_nb.joblib"
        ),
    }
    return vec, models


def explain(model_name, model, vec, x, k=8):
    """Show words that influence the model's prediction."""
    names = np.array(vec.get_feature_names_out())
    idx = x.nonzero()[1]

    if len(idx) == 0:
        return pd.DataFrame(
            columns=["term", "pushes toward", "score"]
        )

    if model_name == "Logistic Regression":
        score = (
            model.coef_[0][idx]
            * x[0, idx].toarray().ravel()
        )
    else:
        score = (
            model.feature_log_prob_[1][idx]
            - model.feature_log_prob_[0][idx]
        ) * x[0, idx].toarray().ravel()

    df = pd.DataFrame({
        "term": names[idx],
        "score": score
    })

    df["pushes toward"] = np.where(
        df["score"] > 0, "Fake", "Real"
    )

    df = df.reindex(
        df["score"].abs().sort_values(ascending=False).index
    ).head(k)

    return df[
        ["term", "pushes toward", "score"]
    ].round(4).reset_index(drop=True)


# Load saved evaluation results
metrics_file = MODELS / "metrics.json"

if not (MODELS / "tfidf_vectorizer.joblib").exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()

try:
    vec, models = load_artifacts()
    info = json.loads(metrics_file.read_text())
    metrics_df = pd.DataFrame(info["metrics"])
except Exception as exc:
    st.error(f"Could not load the saved model or evaluation results: {exc}")
    st.stop()


# Sidebar
with st.sidebar:
    st.title("Settings")

    model_name = st.radio(
        "Select prediction model",
        list(models.keys())
    )

    st.divider()

    st.subheader("Model information")
    st.write(f"**Best model:** {info['best_model']}")
    st.write(f"**Training articles:** {info['n_train']:,}")
    st.write(f"**Testing articles:** {info['n_test']:,}")
    st.write(f"**TF-IDF vocabulary:** {info['vocab_size']:,}")


# Main heading
st.title("📰 Fake News Detection System")
st.caption(
    "An NLP-based application using TF-IDF, "
    "Logistic Regression and Multinomial Naive Bayes."
)

detector_tab, dashboard_tab = st.tabs([
    "News Detector",
    "Project Dashboard"
])


# --------------------------------------------------
# NEWS DETECTOR
# --------------------------------------------------
with detector_tab:
    st.subheader("Analyze a news article")

    choice = st.selectbox(
        "Try a sample article",
        list(EXAMPLES.keys())
    )

    article_text = st.text_area(
        "Enter a news headline and/or article body",
        value=EXAMPLES[choice],
        height=220,
        placeholder="Paste your news article here..."
    )

    if st.button("Analyze News", type="primary"):
        try:
            if not article_text.strip():
                st.warning("Please enter a news article first.")

            else:
                cleaned = clean_text(article_text)

                if len(cleaned.split()) < 5:
                    st.warning(
                        "Please enter at least five meaningful words."
                    )

                else:
                    x = vec.transform([cleaned])

                    if x.nnz == 0:
                        st.warning(
                            "The entered words are not present in "
                            "the model vocabulary. Try a different article."
                        )

                    else:
                        model = models[model_name]
                        p_fake = float(
                            model.predict_proba(x)[0, 1]
                        )

                        confidence = max(p_fake, 1 - p_fake)

                        if p_fake >= 0.5:
                            st.error("Prediction: Likely Fake")
                        else:
                            st.success("Prediction: Likely Real")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Fake probability",
                                f"{p_fake:.2%}"
                            )

                        with col2:
                            st.metric(
                                "Prediction confidence",
                                f"{confidence:.2%}"
                            )

                        st.progress(
                            p_fake,
                            text=f"Probability of Fake: {p_fake:.1%}"
                        )

                        st.subheader("Influential words")
                        st.caption(
                            "These words contributed to the model's "
                            "classification. They are not proof of truth or falsehood."
                        )

                        st.dataframe(
                            explain(model_name, model, vec, x),
                            use_container_width=True,
                            hide_index=True
                        )

        except Exception as exc:
            st.error(
                "An error occurred while analyzing the article. "
                "Please try again with different text."
            )
            st.caption(f"Technical detail: {exc}")

    st.divider()

    st.warning(
        "**Important limitation:** This application identifies patterns "
        "learned from its training dataset. It does not independently "
        "verify facts or check news sources. It may misclassify satire, "
        "breaking news, emerging events, and articles about unfamiliar "
        "topics or writing styles. Always verify important information "
        "using reliable sources."
    )


# --------------------------------------------------
# PROJECT DASHBOARD
# --------------------------------------------------
with dashboard_tab:
    st.header("Project Evaluation Dashboard")

    st.write(
        "Dataset analysis, model performance and evaluation results."
    )

    # Main project statistics
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Raw articles inspected", "44,898")

    with c2:
        st.metric("Training articles", f"{info['n_train']:,}")

    with c3:
        st.metric("Testing articles", f"{info['n_test']:,}")

    with c4:
        st.metric("TF-IDF features", f"{info['vocab_size']:,}")

    st.caption(
        "EDA uses the original 44,898 CSV records. The model uses "
        "38,824 articles after text cleaning and duplicate removal."
    )

    st.divider()

    # Model comparison
    st.subheader("Model Performance Comparison")

    metric_columns = [
        "accuracy", "precision", "recall",
        "f1", "roc_auc"
    ]

    display_df = metrics_df.copy()

    st.dataframe(
        display_df.style.format({
            col: "{:.4f}" for col in metric_columns
        }),
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        metrics_df.set_index("model")[metric_columns]
    )

    st.success(
        f"Selected best model: {info['best_model']}"
    )

    st.divider()

    # Dataset EDA
    st.subheader("Exploratory Data Analysis")

    eda_charts = [
        (
            "label_distribution.png",
            "Dataset Label Distribution"
        ),
        (
            "article_length_distribution.png",
            "Article Length Distribution"
        ),
        (
            "common_words_fake.png",
            "Common Words in Fake News"
        ),
        (
            "common_words_real.png",
            "Common Words in Real News"
        ),
    ]

    for i in range(0, len(eda_charts), 2):
        col1, col2 = st.columns(2)

        for col, (filename, caption) in zip(
            [col1, col2], eda_charts[i:i + 2]
        ):
            image_path = EDA / filename

            with col:
                if image_path.exists():
                    st.image(
                        str(image_path),
                        caption=caption,
                        use_container_width=True
                    )
                else:
                    st.info(f"Chart not found: {filename}")

    inspection_file = EDA / "dataset_inspection.csv"

    if inspection_file.exists():
        st.subheader("Dataset Inspection")
        inspection_df = pd.read_csv(inspection_file)
        st.dataframe(
            inspection_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # Evaluation plots
    st.subheader("Model Evaluation Visualizations")

    confusion_file = REPORTS / "confusion_matrices.png"
    roc_file = REPORTS / "roc_curves.png"

    col1, col2 = st.columns(2)

    with col1:
        if confusion_file.exists():
            st.image(
                str(confusion_file),
                caption="Confusion Matrices",
                use_container_width=True
            )
        else:
            st.info("Confusion matrix image not found.")

    with col2:
        if roc_file.exists():
            st.image(
                str(roc_file),
                caption="ROC Curves",
                use_container_width=True
            )
        else:
            st.info("ROC curve image not found.")

    st.caption(
        "Evaluation metrics are calculated on the held-out test dataset. "
        "Cross-validation scores are calculated on the training dataset."
    )
