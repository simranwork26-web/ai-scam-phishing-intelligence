import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

VERIFIED = "results/fusion/robustness_generated/semantically_verified_59.csv"
SAMPLES = "results/fusion/robustness_sample_100.csv"
MODEL = "results/models/distilbert"
OUT = "results/fusion/robustness_generated/robustness_verified_predictions.csv"

d = pd.read_csv(VERIFIED)
s = pd.read_csv(SAMPLES, usecols=["sample_id", "text"]).rename(columns={"text": "original_text"})
d = d.merge(s, on="sample_id", how="left")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()

texts = d["original_text"].tolist() + d["rewritten_text"].tolist()
scores = []

with torch.no_grad():
    for i in range(0, len(texts), 16):
        batch = tokenizer(
            texts[i:i+16],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        scores.extend(torch.softmax(model(**batch).logits, dim=1)[:, 1].cpu().tolist())

n = len(d)
d["original_score"] = scores[:n]
d["rewritten_score"] = scores[n:]
d["score_delta"] = d["rewritten_score"] - d["original_score"]
d.to_csv(OUT, index=False)

print("Rows:", len(d))
print("Device:", device)
print("Saved:", OUT)
print(d.groupby("transformation_type")["score_delta"].agg(["count","mean","median","min","max"]).to_string())
