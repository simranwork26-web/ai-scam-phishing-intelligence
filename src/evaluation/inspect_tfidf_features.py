from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "results/models"


def main() -> None:
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODEL_DIR / "tfidf_logistic_regression.joblib")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = model.coef_[0]

    positive_idx = np.argsort(coefficients)[-30:][::-1]
    negative_idx = np.argsort(coefficients)[:30]

    print("=== TF-IDF Feature Inspection ===")
    print()
    print("Top features associated with class 1:")
    for idx in positive_idx:
        print(f"{feature_names[idx]:40s} {coefficients[idx]: .6f}")

    print()
    print("Top features associated with class 0:")
    for idx in negative_idx:
        print(f"{feature_names[idx]:40s} {coefficients[idx]: .6f}")


if __name__ == "__main__":
    main()
