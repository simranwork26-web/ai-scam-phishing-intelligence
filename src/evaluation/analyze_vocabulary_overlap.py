from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/processed/text_splits/train.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    ling = df.loc[df["source"] == "Ling", "text"]
    spamassasin = df.loc[df["source"] == "SpamAssasin", "text"]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    vectorizer.fit(pd.concat([ling, spamassasin]))

    ling_vocab = set(
        vectorizer.transform(ling).nonzero()[1]
    )
    spamassasin_vocab = set(
        vectorizer.transform(spamassasin).nonzero()[1]
    )

    shared = ling_vocab & spamassasin_vocab

    print("=== Vocabulary Overlap ===")
    print(f"Ling vocabulary:        {len(ling_vocab)}")
    print(f"SpamAssasin vocabulary: {len(spamassasin_vocab)}")
    print(f"Shared vocabulary:      {len(shared)}")
    print()

    print(
        "Shared / Ling:        "
        f"{len(shared) / len(ling_vocab):.4f}"
    )
    print(
        "Shared / SpamAssasin: "
        f"{len(shared) / len(spamassasin_vocab):.4f}"
    )


if __name__ == "__main__":
    main()
