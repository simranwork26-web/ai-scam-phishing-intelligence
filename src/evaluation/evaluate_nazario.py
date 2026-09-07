from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NAZARIO_PATH = ROOT / "data/raw/text/Nazario.csv"
MODEL_DIR = ROOT / "results/models"
OUTPUT_PATH = ROOT / "results/dataset_investigation/nazario_predictions.csv"


def main() -> None:
    df = pd.read_csv(
        NAZARIO_PATH,
        usecols=["subject", "body", "label"],
    )

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = (
        df["subject"].str.strip() + " " + df["body"].str.strip()
    ).str.strip()

    vectorizer = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )
    model = joblib.load(
        MODEL_DIR / "tfidf_logistic_regression.joblib"
    )

    probabilities = model.predict_proba(
        vectorizer.transform(df["text"])
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    df["prediction"] = predictions
    df["probability_class_1"] = probabilities

    print("=== External Evaluation: Nazario ===")
    print(f"Samples: {len(df)}")
    print(f"True labels: {df['label'].unique().tolist()}")
    print()
    print("Predicted class distribution:")
    print(df["prediction"].value_counts().sort_index())
    print()
    print(
        "Correctly identified phishing:",
        int((df["prediction"] == 1).sum()),
        "/",
        len(df),
    )
    print(
        "Phishing recall:",
        round((df["prediction"] == 1).mean(), 4),
    )
    print()
    print("Probability summary:")
    print(df["probability_class_1"].describe())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[[
        "text",
        "label",
        "prediction",
        "probability_class_1",
    ]].to_csv(OUTPUT_PATH, index=False)

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
