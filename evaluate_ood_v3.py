import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

df = pd.read_csv("data/processed/frozen/ood_v3_frozen.csv")
model_path = "results/models/distilbert_short_benign"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device)
model.eval()

texts = df["text"].fillna("").astype(str).tolist()
probabilities = []

batch_size = 16

for start in range(0, len(texts), batch_size):
    batch_texts = texts[start:start + batch_size]

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]

    probabilities.extend(
        probs.cpu().numpy().tolist()
    )

y_true = (df["label"] == "spam").astype(int).to_numpy()
probabilities = np.array(probabilities)
y_pred = (probabilities >= 0.5).astype(int)

print("Model D OOD v3")
print("Samples:", len(df))
print("Device:", device)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, probabilities))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
