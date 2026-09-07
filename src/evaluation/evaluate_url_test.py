from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data/processed/url_domain_splits/test.csv"
MODEL_DIR = ROOT / "results/models"

FEATURE_COLUMNS = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "slash_count",
    "question_mark_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "at_count",
    "digit_count",
    "special_char_count",
    "subdomain_count",
    "has_ip_hostname",
    "has_port",
    "has_query",
    "has_fragment",
    "has_encoded_chars",
    "suspicious_token_count",
]


def main():
    test = pd.read_csv(TEST_PATH)

    X_test = test[FEATURE_COLUMNS]
    y_test = (test["label"] == "bad").astype(int)

    scaler = joblib.load(
        MODEL_DIR / "url_feature_scaler.joblib"
    )
    model = joblib.load(
        MODEL_DIR / "url_logistic_regression.joblib"
    )

    X_test_scaled = scaler.transform(X_test)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("=== URL Logistic Regression: Domain-Aware Test ===")
    print(f"Test samples: {len(test)}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions):.4f}")
    print(f"F1:        {f1_score(y_test, predictions):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")
    print()
    print(classification_report(y_test, predictions, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
