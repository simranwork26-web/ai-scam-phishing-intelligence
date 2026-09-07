import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

INPUT_PATH = "data/raw/url/phiusiil.csv"
OUTPUT_DIR = "data/processed/phiusiil_domain_splits"

URL_FEATURES = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]

df = pd.read_csv(INPUT_PATH)

# PhiUSIIL mapping confirmed from the local data:
# 0 = phishing, 1 = legitimate
df = df[["URL", "Domain", "label"] + URL_FEATURES].copy()

# Use a stable normalized grouping key.
df["group_domain"] = (
    df["Domain"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

# Empty domains cannot provide meaningful grouping information.
# Keep them as a single group rather than silently dropping data.
df["group_domain"] = df["group_domain"].replace("", "__EMPTY_DOMAIN__")

gss_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, temp_idx = next(
    gss_1.split(df, y=df["label"], groups=df["group_domain"])
)

train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

gss_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)

val_idx, test_idx = next(
    gss_2.split(temp, y=temp["label"], groups=temp["group_domain"])
)

val = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

for split in (train, val, test):
    split.drop(columns=["group_domain"], inplace=True)

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
val.to_csv(f"{OUTPUT_DIR}/validation.csv", index=False)
test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)

print("PhiUSIIL domain-aware split created.")
print(f"Train:      {train.shape}")
print(f"Validation: {val.shape}")
print(f"Test:       {test.shape}")
print(f"Total:      {len(train) + len(val) + len(test)}")

print("\nLabel counts:")
for name, split in [
    ("train", train),
    ("validation", val),
    ("test", test),
]:
    print(
        name,
        split["label"].value_counts().sort_index().to_dict(),
        "phishing_rate=",
        round((split["label"] == 0).mean(), 4),
    )

print("\nDomain overlap:")
train_domains = set(train["Domain"].astype(str).str.lower())
val_domains = set(val["Domain"].astype(str).str.lower())
test_domains = set(test["Domain"].astype(str).str.lower())

print("Train ∩ Validation:", len(train_domains & val_domains))
print("Train ∩ Test:", len(train_domains & test_domains))
print("Validation ∩ Test:", len(val_domains & test_domains))
