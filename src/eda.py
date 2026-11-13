
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load and combine Fake and Real news datasets."""
    fake_path = DATA_DIR / "Fake.csv"
    real_path = DATA_DIR / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            "Fake.csv or True.csv is missing from the data folder."
        )

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake["label"] = "Fake"
    real["label"] = "Real"

    return pd.concat([fake, real], ignore_index=True)


def clean_text(text):
    """Basic text cleaning for word-frequency analysis."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text


def run_eda():
    df = load_dataset()

    # 1. Dataset inspection
    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    # Prepare article text
    df["text"] = df["text"].fillna("")
    df["article_length"] = df["text"].apply(
        lambda text: len(str(text).split())
    )

    # 2. Label distribution
    plt.figure(figsize=(7, 5))
    df["label"].value_counts().plot(kind="bar")
    plt.title("Fake vs Real News Distribution")
    plt.xlabel("News Category")
    plt.ylabel("Number of Articles")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png")
    plt.close()

    # 3. Article length distribution
    plt.figure(figsize=(8, 5))
    for label in ["Fake", "Real"]:
        df.loc[df["label"] == label, "article_length"].clip(
            upper=2000
        ).plot(
            kind="hist",
            bins=40,
            alpha=0.5,
            label=label
        )

    plt.title("Article Length Distribution")
    plt.xlabel("Number of Words (capped at 2000)")
    plt.ylabel("Number of Articles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "article_length_distribution.png")
    plt.close()

    # 4. Common words by category
    stop_words = {
        "the", "and", "for", "that", "this", "with", "from",
        "was", "were", "are", "has", "have", "had", "not",
        "but", "you", "your", "they", "their", "will", "would",
        "about", "after", "before", "into", "than", "then",
        "said", "also", "just", "more", "some", "what", "when",
        "who", "how", "why", "all", "can", "its", "our",
        "his", "her", "she", "him", "been", "being", "one",
        "two", "out", "over", "new", "could", "which"
    }

    for label in ["Fake", "Real"]:
        words = []

        for article in df.loc[df["label"] == label, "text"]:
            tokens = clean_text(article).split()
            words.extend(
                word for word in tokens
                if len(word) > 2 and word not in stop_words
            )

        common_words = Counter(words).most_common(15)

        words_df = pd.DataFrame(
            common_words,
            columns=["Word", "Frequency"]
        )

        plt.figure(figsize=(9, 6))
        plt.barh(
            words_df["Word"][::-1],
            words_df["Frequency"][::-1]
        )
        plt.title(f"Top 15 Common Words in {label} News")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / f"common_words_{label.lower()}.png"
        )
        plt.close()

    # 5. Save dataset inspection summary
    inspection = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values
    })

    inspection.to_csv(
        OUTPUT_DIR / "dataset_inspection.csv",
        index=False
    )

    print("\nEDA completed successfully!")
    print("Charts and inspection file saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
