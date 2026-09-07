import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/processed/url_dataset.csv"
OUTPUT_DIR = ROOT / "data/processed/url_domain_splits"

RANDOM_STATE = 42


def main():
    df = pd.read_csv(INPUT_PATH)

    # Hostname was already validated with the safe parser during
    # the leakage investigation. Use the stored URL strings here
    # and derive a conservative grouping key from the feature data.
    from urllib.parse import urlparse

    def safe_hostname(url):
        try:
            url = str(url).strip()
            if "://" not in url:
                url = "http://" + url
            return (urlparse(url).hostname or "").lower()
        except (ValueError, TypeError):
            return ""

    df["hostname"] = df["url"].map(safe_hostname)

    # Give malformed URLs unique deterministic groups so they cannot
    # leak across splits, while retaining them in the dataset.
    malformed = df["hostname"].eq("")
    df.loc[malformed, "hostname"] = df.loc[malformed].index.map(lambda x: "MALFORMED_" + str(x))

    # First: 80% train, 20% temporary.
    groups = df["hostname"]

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        splitter_1.split(df, groups=groups)
    )

    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    # Second: split temporary 50/50 into validation and test.
    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = next(
        splitter_2.split(temp, groups=temp["hostname"])
    )

    val = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()

    # Remove grouping column from saved datasets.
    for split in (train, val, test):
        split.drop(columns=["hostname"], inplace=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Domain-aware URL split created.")
    print()

    for name, split in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        bad_rate = (split["label"] == "bad").mean()

        print(
            f"{name}: {len(split)} rows | "
            f"good={sum(split['label'] == 'good')} | "
            f"bad={sum(split['label'] == 'bad')} | "
            f"bad rate={bad_rate:.4f}"
        )

    print()
    print("Total:", len(train) + len(val) + len(test))
    print("Original:", len(df))


if __name__ == "__main__":
    main()
