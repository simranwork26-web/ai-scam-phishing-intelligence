from pathlib import Path

import joblib
import pandas as pd

from src.features.manipulation_signals import detect_manipulation_signals


ROOT = Path(__file__).resolve().parents[2]

TEXT_TEST = ROOT / "data/processed/text_splits/test.csv"
URL_TEST = ROOT / "data/processed/url_domain_splits/test.csv"

URL_MODEL_DIR = ROOT / "results/models"

OUTPUT_DIR = ROOT / "results/fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


URL_FEATURE_COLUMNS = [
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
    print("Loading test datasets...")

    text = pd.read_csv(TEXT_TEST)
    urls = pd.read_csv(URL_TEST)

    print(f"Text test rows: {len(text)}")
    print(f"URL test rows:  {len(urls)}")

    # ------------------------------------------------------------------
    # URL evidence
    # ------------------------------------------------------------------
    scaler = joblib.load(
        URL_MODEL_DIR / "url_feature_scaler.joblib"
    )

    url_model = joblib.load(
        URL_MODEL_DIR / "url_logistic_regression.joblib"
    )

    X_url = urls[URL_FEATURE_COLUMNS]
    X_url = scaler.transform(X_url)

    url_probability = url_model.predict_proba(X_url)[:, 1]

    url_evidence = pd.DataFrame(
        {
            "url": urls["url"],
            "url_probability": url_probability,
            "url_label": urls["label"],
        }
    )

    # ------------------------------------------------------------------
    # Manipulation signals
    # ------------------------------------------------------------------
    signals = text["text"].fillna("").map(
        detect_manipulation_signals
    )

    signal_df = pd.DataFrame(
        {
            "urgency": signals.map(lambda x: int(x["urgency"])),
            "financial": signals.map(lambda x: int(x["financial"])),
            "authority": signals.map(lambda x: int(x["authority"])),
            "signal_count": signals.map(
                lambda x: sum(
                    [
                        x["urgency"],
                        x["financial"],
                        x["authority"],
                    ]
                )
            ),
            "combination_count": signals.map(
                lambda x: len(x["combinations"])
            ),
        }
    )

    # ------------------------------------------------------------------
    # Save text test rows + manipulation evidence.
    #
    # Text probabilities are added later from the DistilBERT evaluator.
    # Keeping this intermediate artifact makes the fusion pipeline
    # reproducible and prevents silently inventing alignment.
    # ------------------------------------------------------------------
    evidence = pd.concat(
        [
            text.reset_index(drop=True),
            signal_df.reset_index(drop=True),
        ],
        axis=1,
    )

    evidence.to_csv(
        OUTPUT_DIR / "text_manipulation_evidence.csv",
        index=False,
    )

    url_evidence.to_csv(
        OUTPUT_DIR / "url_evidence.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(OUTPUT_DIR / "text_manipulation_evidence.csv")
    print(OUTPUT_DIR / "url_evidence.csv")


if __name__ == "__main__":
    main()
