import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/url_dataset.csv"

def safe_hostname(url):
    try:
        url = str(url).strip()
        if "://" not in url:
            url = "http://" + url
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()
    except (ValueError, TypeError):
        return ""

def main():
    df = pd.read_csv(DATA_PATH)

    df["hostname"] = df["url"].map(safe_hostname)

    print("Rows:", len(df))
    print("Unique hostnames:", df["hostname"].nunique())
    print(
        "Malformed/unparseable URLs:",
        (df["hostname"] == "").sum()
    )

    hostname_label_counts = (
        df.groupby("hostname")["label"]
        .nunique()
    )

    print(
        "Hostnames with both labels:",
        (hostname_label_counts == 2).sum()
    )

    print("\nTop hostnames by URL count:")
    print(
        df["hostname"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nLabel distribution by hostname:")
    print(
        hostname_label_counts
        .value_counts()
        .sort_index()
        .to_string()
    )

if __name__ == "__main__":
    main()
