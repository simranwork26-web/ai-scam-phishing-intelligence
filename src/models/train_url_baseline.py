from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/url_domain_splits/train.csv"
VAL_PATH = ROOT / "data/processed/url_domain_splits/val.csv"
MODEL_DIR = ROOT / "results/models"

TARGET = "label"

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


def evaluate(split_name, y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    print(f"=== URL Logistic Regression: {split_name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.4f}")
    print(f"Precision: {precision_score(y_true, predictions):.4f}")
    print(f"Recall:    {recall_score(y_true, predictions):.4f}")
    print(f"F1:        {f1_score(y_true, predictions):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, probabilities):.4f}")
    print()
    print(classification_report(y_true, predictions, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))
    print()


def main():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    X_train = train[FEATURE_COLUMNS]
    X_val = val[FEATURE_COLUMNS]

    y_train = (train[TARGET] == "bad").astype(int)
    y_val = (val[TARGET] == "bad").astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train_scaled, y_train)

    train_probabilities = model.predict_proba(X_train_scaled)[:, 1]
    val_probabilities = model.predict_proba(X_val_scaled)[:, 1]

    evaluate("Validation", y_val, val_probabilities)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        scaler,
        MODEL_DIR / "url_feature_scaler.joblib",
    )
    joblib.dump(
        model,
        MODEL_DIR / "url_logistic_regression.joblib",
    )

    print("Saved:")
    print(MODEL_DIR / "url_feature_scaler.joblib")
    print(MODEL_DIR / "url_logistic_regression.joblib")


if __name__ == "__main__":
    main()
