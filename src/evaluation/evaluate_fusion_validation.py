import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


PATH = "results/fusion/final_val_evidence.csv"


def evaluate(name, y, probability):
    prediction = (probability >= 0.5).astype(int)

    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y, prediction):.4f}")
    print(f"Precision: {precision_score(y, prediction):.4f}")
    print(f"Recall:    {recall_score(y, prediction):.4f}")
    print(f"F1:        {f1_score(y, prediction):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, probability):.4f}")


def main():
    df = pd.read_csv(PATH)

    y = df["label"].astype(int)

    text = df["text_probability"].astype(float)
    url = df["url_max_probability"].astype(float)

    signals = (
        df["urgency"]
        + df["financial"]
        + df["authority"]
    )

    signal_score = signals / 3.0

    # Reference: text model alone.
    evaluate(
        "DistilBERT",
        y,
        text,
    )

    # Strategy 1:
    # Text dominates; URL is weak secondary evidence.
    fusion_1 = (
        0.90 * text
        + 0.10 * url
    )

    evaluate(
        "Fusion 1: 90% text + 10% URL",
        y,
        fusion_1,
    )

    # Strategy 2:
    # Text + manipulation evidence.
    fusion_2 = (
        0.90 * text
        + 0.10 * signal_score
    )

    evaluate(
        "Fusion 2: 90% text + 10% signals",
        y,
        fusion_2,
    )

    # Strategy 3:
    # Text remains dominant, with both secondary evidence sources.
    fusion_3 = (
        0.80 * text
        + 0.10 * url
        + 0.10 * signal_score
    )

    evaluate(
        "Fusion 3: 80% text + 10% URL + 10% signals",
        y,
        fusion_3,
    )


if __name__ == "__main__":
    main()
