from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

LING_PATH = ROOT / "data/raw/text/Ling.csv"
SPAMASSASIN_PATH = ROOT / "data/raw/text/SpamAssasin.csv"
OUTPUT_PATH = ROOT / "data/processed/text_dataset.csv"


def normalize_for_dedup(text: pd.Series) -> pd.Series:
    return (
        text.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
    )


def load_dataset(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["subject", "body", "label"])

    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["source"] = source

    return df


def main() -> None:
    ling = load_dataset(LING_PATH, "Ling")
    spamassasin = load_dataset(SPAMASSASIN_PATH, "SpamAssasin")

    df = pd.concat([ling, spamassasin], ignore_index=True)

    before_dedup = len(df)

    dedup_text = normalize_for_dedup(
        df["subject"] + " " + df["body"]
    )

    df = df.loc[~dedup_text.duplicated()].copy()

    df["text"] = (
        df["subject"].str.strip()
        + " "
        + df["body"].str.strip()
    ).str.strip()

    df = df[
        ["text", "subject", "body", "label", "source"]
    ].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=== Text Dataset Preparation ===")
    print(f"Rows before deduplication: {before_dedup}")
    print(f"Rows after deduplication:  {len(df)}")
    print(f"Rows removed:              {before_dedup - len(df)}")
    print()
    print("Label distribution:")
    print(df["label"].value_counts().sort_index())
    print()
    print("Source distribution:")
    print(df["source"].value_counts())
    print()
    print("Empty final text rows:", df["text"].eq("").sum())
    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
