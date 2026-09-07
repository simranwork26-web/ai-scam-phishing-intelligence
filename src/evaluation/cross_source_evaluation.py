from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/text_splits"

RANDOM_STATE = 42


def evaluate(train_source: str, test_source: str) -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_df = train_df[train_df["source"] == train_source].copy()
    test_df = test_df[test_df["source"] == test_source].copy()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, train_df["label"])

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n=== Train: {train_source} | Test: {test_source} ===")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples:     {len(test_df)}")
    print(f"TF-IDF features:  {X_train.shape[1]}")
    print()
    print(classification_report(test_df["label"], predictions, digits=4))
    print(f"ROC-AUC: {roc_auc_score(test_df['label'], probabilities):.4f}")


def main() -> None:
    evaluate("Ling", "SpamAssasin")
    evaluate("SpamAssasin", "Ling")


if __name__ == "__main__":
    main()
