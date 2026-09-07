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
        return (urlparse(url).hostname or "").lower()
    except (ValueError, TypeError):
        return ""

def main():
    df = pd.read_csv(DATA_PATH)
    df["hostname"] = df["url"].map(safe_hostname)

    label_counts = df.groupby("hostname")["label"].nunique()
    mixed_hosts = label_counts[label_counts == 2].index

    mixed = df[df["hostname"].isin(mixed_hosts)]

    print("Mixed-label hostnames:", len(mixed_hosts))
    print("URLs belonging to mixed-label hostnames:", len(mixed))
    print("\nLabels within mixed-label hostnames:")
    print(mixed.groupby(["hostname", "label"]).size().to_string())

if __name__ == "__main__":
    main()
