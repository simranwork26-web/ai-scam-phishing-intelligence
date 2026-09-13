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

DATA_PATH = "data/processed/frozen/ood_v3_frozen.csv"

df = pd.read_csv(DATA_PATH)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

texts = df["text"].fillna("").astype(str).tolist()
y_true = (df["label"] == "spam").astype(int).to_numpy()

models = {
    "Original": "results/models/distilbert",
    "Model_D": "results/models/distilbert_short_benign",
}

all_results = {}

for name, model_path in models.items():
    print(f"\nLoading {name}: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(device)
    model.eval()

    probabilities = []

    for start in range(0, len(texts), 16):
        batch = texts[start:start + 16]

        encoded = tokenizer(
            batch,
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

    probabilities = np.array(probabilities)
    predictions = (probabilities >= 0.5).astype(int)

    all_results[name] = {
        "probabilities": probabilities,
        "predictions": predictions,
    }

    print(name)
    print("Accuracy:", accuracy_score(y_true, predictions))
    print("Precision:", precision_score(y_true, predictions))
    print("Recall:", recall_score(y_true, predictions))
    print("F1:", f1_score(y_true, predictions))
    print("ROC-AUC:", roc_auc_score(y_true, probabilities))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))

out = df[
    ["scenario_id", "label", "text"]
].copy()

out["original_probability"] = all_results[
    "Original"
]["probabilities"]

out["model_d_probability"] = all_results[
    "Model_D"
]["probabilities"]

out["original_prediction"] = all_results[
    "Original"
]["predictions"]

out["model_d_prediction"] = all_results[
    "Model_D"
]["predictions"]

out["prediction_changed"] = (
    out["original_prediction"]
    != out["model_d_prediction"]
)

output_path = (
    "results/dataset_investigation/"
    "ood_v3_original_vs_model_d.csv"
)

out.to_csv(output_path, index=False)

print(
    "\nPrediction changes:",
    int(out["prediction_changed"].sum())
)

print("Saved:", output_path)
