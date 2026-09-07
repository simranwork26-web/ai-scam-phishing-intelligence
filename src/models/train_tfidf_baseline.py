from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/text_splits"
MODEL_DIR = ROOT / "results/models"

RANDOM_STATE = 42


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_validation = vectorizer.transform(validation_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    y_train = train_df["label"]
    y_validation = validation_df["label"]
    y_test = test_df["label"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    validation_pred = model.predict(X_validation)
    validation_prob = model.predict_proba(X_validation)[:, 1]

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    print("=== TF-IDF + Logistic Regression Baseline ===")
    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples:       {len(test_df)}")
    print(f"TF-IDF features:    {X_train.shape[1]}")
    print()

    print("Validation ROC-AUC:", round(roc_auc_score(y_validation, validation_prob), 4))
    print()
    print("Validation classification report:")
    print(classification_report(y_validation, validation_pred, digits=4))
    print("Validation confusion matrix:")
    print(confusion_matrix(y_validation, validation_pred))
    print()

    print("Test ROC-AUC:", round(roc_auc_score(y_test, test_prob), 4))
    print()
    print("Test classification report:")
    print(classification_report(y_test, test_pred, digits=4))
    print("Test confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(model, MODEL_DIR / "tfidf_logistic_regression.joblib")

    print()
    print("Saved:")
    print(MODEL_DIR / "tfidf_vectorizer.joblib")
    print(MODEL_DIR / "tfidf_logistic_regression.joblib")


if __name__ == "__main__":
    main()
