from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data/processed/text_dataset.csv"
OUTPUT_DIR = ROOT / "data/processed/text_splits"

RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    stratify_key = df["source"].astype(str) + "_" + df["label"].astype(str)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify_key,
    )

    temp_stratify_key = (
        temp_df["source"].astype(str) + "_" + temp_df["label"].astype(str)
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_stratify_key,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("=== Text Dataset Split ===")
    print(f"Total rows:       {len(df)}")
    print(f"Training rows:    {len(train_df)}")
    print(f"Validation rows:  {len(val_df)}")
    print(f"Test rows:        {len(test_df)}")
    print()

    print("Overall label distribution:")
    print(df["label"].value_counts(normalize=True).sort_index())
    print()

    print("Training label distribution:")
    print(train_df["label"].value_counts(normalize=True).sort_index())
    print()

    print("Validation label distribution:")
    print(val_df["label"].value_counts(normalize=True).sort_index())
    print()

    print("Test label distribution:")
    print(test_df["label"].value_counts(normalize=True).sort_index())
    print()

    print("Training source distribution:")
    print(train_df["source"].value_counts())
    print()

    print("Validation source distribution:")
    print(val_df["source"].value_counts())
    print()

    print("Test source distribution:")
    print(test_df["source"].value_counts())


if __name__ == "__main__":
    main()
