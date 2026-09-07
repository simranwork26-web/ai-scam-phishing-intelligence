import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "results/fusion/text_val_evidence.csv"
OUTPUT_PATH = ROOT / "results/fusion/text_val_with_urls.csv"

URL_PATTERN = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"]+"
)


def clean_url(url):
    url = url.rstrip(".,;:!?)]}>\"'")
    return url


def main():
    df = pd.read_csv(INPUT_PATH)

    df["extracted_urls"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .map(
            lambda text: [
                clean_url(url)
                for url in URL_PATTERN.findall(text)
            ]
        )
    )

    df["url_count"] = df["extracted_urls"].map(len)

    df.to_csv(OUTPUT_PATH, index=False)

    print("URL extraction complete.")
    print(f"Rows: {len(df)}")
    print(f"Messages containing URLs: {(df['url_count'] > 0).sum()}")
    print(f"Messages without URLs: {(df['url_count'] == 0).sum()}")
    print(f"Total extracted URL occurrences: {df['url_count'].sum()}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
