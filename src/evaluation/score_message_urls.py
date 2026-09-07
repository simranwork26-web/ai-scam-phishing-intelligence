from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd

from src.features.url_features import extract_url_features


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "results/fusion/text_val_with_urls.csv"
OUTPUT_PATH = ROOT / "results/fusion/final_val_evidence.csv"

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
    df = pd.read_csv(INPUT_PATH)

    scaler = joblib.load(
        MODEL_DIR / "url_feature_scaler.joblib"
    )
    model = joblib.load(
        MODEL_DIR / "url_logistic_regression.joblib"
    )

    url_max = []
    url_mean = []
    url_counts = []

    for urls in df["extracted_urls"].fillna("[]"):
        # The CSV stores Python-list strings.
        try:
            parsed_urls = eval(urls)
        except Exception:
            parsed_urls = []

        probabilities = []

        for url in parsed_urls:
            try:
                features = extract_url_features(url)
                row = pd.DataFrame([features])[FEATURE_COLUMNS]
                scaled = scaler.transform(row)
                probability = model.predict_proba(scaled)[0, 1]
                probabilities.append(float(probability))
            except Exception:
                continue

        url_counts.append(len(probabilities))

        if probabilities:
            url_max.append(max(probabilities))
            url_mean.append(float(np.mean(probabilities)))
        else:
            url_max.append(0.0)
            url_mean.append(0.0)

    result = df.copy()
    result["url_count_scored"] = url_counts
    result["url_max_probability"] = url_max
    result["url_mean_probability"] = url_mean

    result.to_csv(OUTPUT_PATH, index=False)

    print("Message-level URL scoring complete.")
    print(f"Rows: {len(result)}")
    print(
        "Messages with scored URLs:",
        (result["url_count_scored"] > 0).sum(),
    )
    print(
        "Messages without scored URLs:",
        (result["url_count_scored"] == 0).sum(),
    )
    print(
        "Maximum URL probability:",
        f"{result['url_max_probability'].max():.6f}",
    )
    print(
        "Mean URL probability:",
        f"{result['url_mean_probability'].mean():.6f}",
    )
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
