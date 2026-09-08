from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data/processed/text_splits/test.csv"
MODEL_PATH = ROOT / "results/models/distilbert_augmented"

MAX_LENGTH = 256
BATCH_SIZE = 16


def main() -> None:
    df = pd.read_csv(TEST_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()

    texts = df["text"].fillna("").astype(str).tolist()
    labels = df["label"].to_numpy()

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

    output = df[["text", "subject", "source", "label"]].copy()
    output["probability"] = probabilities
    output["prediction"] = predictions
    output["correct"] = output["label"] == output["prediction"]
    output_path = ROOT / "results/dataset_investigation/distilbert_augmented_test_predictions.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")

    print("=== Augmented DistilBERT Test Evaluation ===")
    print(f"Test samples: {len(df)}")
    print(f"Device: {device}")
    print()

    print(f"Accuracy:  {accuracy_score(labels, predictions):.4f}")
    print(f"Precision: {precision_score(labels, predictions):.4f}")
    print(f"Recall:    {recall_score(labels, predictions):.4f}")
    print(f"F1:        {f1_score(labels, predictions):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(labels, probabilities):.4f}")
    print()

    print("Classification report:")
    print(classification_report(labels, predictions, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))


if __name__ == "__main__":
    main()
