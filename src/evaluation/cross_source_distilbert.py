from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/text_splits"
MODEL_DIR = ROOT / "results/models"

MAX_LENGTH = 256
BATCH_SIZE = 16


def evaluate(train_source: str, test_source: str) -> None:
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df = test_df[test_df["source"] == test_source].copy()

    model_path = MODEL_DIR / f"distilbert_cross_source_{train_source.lower()}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    texts = test_df["text"].fillna("").astype(str).tolist()
    labels = test_df["label"].to_numpy()

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

    print()
    print(f"=== DistilBERT: Train {train_source} | Test {test_source} ===")
    print(f"Training source: {train_source}")
    print(f"Test source:     {test_source}")
    print(f"Test samples:    {len(test_df)}")
    print(f"Device:          {device}")
    print()

    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(f"Precision: {precision_score(labels, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, predictions, zero_division=0):.4f}")
    print(f"F1:        {f1_score(labels, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(labels, probabilities):.4f}")
    print()

    print("Classification report:")
    print(classification_report(labels, predictions, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))


def main() -> None:
    evaluate("Ling", "SpamAssasin")
    evaluate("SpamAssasin", "Ling")


if __name__ == "__main__":
    main()
