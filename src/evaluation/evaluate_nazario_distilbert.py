from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
NAZARIO_PATH = ROOT / "data/raw/text/Nazario.csv"
MODEL_PATH = ROOT / "results/models/distilbert"

MAX_LENGTH = 256
BATCH_SIZE = 16


def main() -> None:
    df = pd.read_csv(NAZARIO_PATH)

    texts = (
        df["subject"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    ).str.strip().tolist()

    labels = df["label"].to_numpy()

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

    correct = (predictions == labels).sum()
    missed = (predictions != labels).sum()

    print("=== DistilBERT External Nazario Evaluation ===")
    print(f"Nazario samples:       {len(df)}")
    print(f"Unique labels:         {np.unique(labels)}")
    print(f"Device:                {device}")
    print()
    print(f"Predicted class 0:     {(predictions == 0).sum()}")
    print(f"Predicted class 1:     {(predictions == 1).sum()}")
    print()
    print(f"Correctly identified:  {correct}")
    print(f"Missed:                {missed}")
    print(f"Phishing recall:       {correct / len(df):.4f}")
    print()
    print("Probability summary:")
    print(f"Mean:                  {probabilities.mean():.6f}")
    print(f"Std:                   {probabilities.std():.6f}")
    print(f"Minimum:               {probabilities.min():.6f}")
    print(f"25th percentile:       {np.percentile(probabilities, 25):.6f}")
    print(f"Median:                {np.median(probabilities):.6f}")
    print(f"75th percentile:       {np.percentile(probabilities, 75):.6f}")
    print(f"Maximum:               {probabilities.max():.6f}")


if __name__ == "__main__":
    main()
