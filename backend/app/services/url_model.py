from pathlib import Path

import joblib
import pandas as pd

from src.features.url_features import extract_url_features


ROOT = Path(__file__).resolve().parents[3]
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


class URLModel:
    def __init__(self):
        self.scaler = joblib.load(
            MODEL_DIR / "url_feature_scaler.joblib"
        )
        self.model = joblib.load(
            MODEL_DIR / "url_logistic_regression.joblib"
        )

    def predict(self, url: str) -> float:
        features = extract_url_features(url)
        row = pd.DataFrame([features])[FEATURE_COLUMNS]
        scaled = self.scaler.transform(row)
        return float(self.model.predict_proba(scaled)[0, 1])


_model = None


def get_url_model():
    global _model

    if _model is None:
        _model = URLModel()

    return _model
