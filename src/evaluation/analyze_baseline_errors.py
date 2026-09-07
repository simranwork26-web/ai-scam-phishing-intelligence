from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/text_splits/test.csv"
MODEL_DIR = ROOT / "results/models"
OUTPUT_DIR = ROOT / "results/dataset_investigation"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "tfidf_logistic_regression.joblib")

    X_test = vectorizer.transform(df["text"])
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    df["prediction"] = predictions
    df["probability_class_1"] = probabilities

    false_positives = df[
        (df["label"] == 0) & (df["prediction"] == 1)
    ].copy()

    false_negatives = df[
        (df["label"] == 1) & (df["prediction"] == 0)
    ].copy()

    false_positives = false_positives.sort_values(
        "probability_class_1", ascending=False
    )

    false_negatives = false_negatives.sort_values(
        "probability_class_1", ascending=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    false_positives.to_csv(
        OUTPUT_DIR / "baseline_false_positives.csv", index=False
    )

    false_negatives.to_csv(
        OUTPUT_DIR / "baseline_false_negatives.csv", index=False
    )

    print("=== Baseline Error Analysis ===")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")
    print()

    print("False positives by source:")
    print(false_positives["source"].value_counts())
    print()

    print("False negatives by source:")
    print(false_negatives["source"].value_counts())
    print()

    print("Saved:")
    print(OUTPUT_DIR / "baseline_false_positives.csv")
    print(OUTPUT_DIR / "baseline_false_negatives.csv")


if __name__ == "__main__":
    main()
