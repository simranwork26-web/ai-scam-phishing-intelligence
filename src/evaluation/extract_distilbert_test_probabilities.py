from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data/processed/text_splits/test.csv"
MODEL_PATH = ROOT / "results/models/distilbert"
OUTPUT_DIR = ROOT / "results/fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 256
BATCH_SIZE = 16


def main():
    df = pd.read_csv(TEST_PATH)

    texts = df["text"].fillna("").astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    probabilities = []

    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
            probabilities.extend(
                probs.detach().cpu().numpy().tolist()
            )

    result = df.copy()
    result["text_probability"] = np.asarray(probabilities)

    output_path = OUTPUT_DIR / "text_test_evidence.csv"
    result.to_csv(output_path, index=False)

    print("DistilBERT test probabilities extracted.")
    print(f"Rows: {len(result)}")
    print(
        "Probability range:",
        f"{result['text_probability'].min():.6f}",
        "to",
        f"{result['text_probability'].max():.6f}",
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
