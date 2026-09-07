from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
NAZARIO_PATH = ROOT / "data/raw/text/Nazario.csv"
MODEL_PATH = ROOT / "results/models/distilbert"
OUTPUT_PATH = ROOT / "results/dataset_investigation/nazario_distilbert_false_negatives.csv"

MAX_LENGTH = 256
BATCH_SIZE = 16


def main() -> None:
    df = pd.read_csv(NAZARIO_PATH)

    texts = (
        df["subject"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    ).str.strip().tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            probabilities.extend(probs.cpu().numpy())

    probabilities = np.array(probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    df["predicted_probability"] = probabilities
    df["predicted_label"] = predictions

    false_negatives = df[
        (df["label"] == 1) & (df["predicted_label"] == 0)
    ].copy()

    false_negatives = false_negatives.sort_values(
        "predicted_probability"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    false_negatives.to_csv(OUTPUT_PATH, index=False)

    print("=== DistilBERT Nazario Error Analysis ===")
    print(f"Total Nazario samples: {len(df)}")
    print(f"False negatives:       {len(false_negatives)}")
    print(f"Saved to:              {OUTPUT_PATH}")
    print()

    print("False-negative probability summary:")
    print(false_negatives["predicted_probability"].describe())

    print()
    print("Lowest-confidence false negatives:")
    print(
        false_negatives[
            ["subject", "predicted_probability"]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
