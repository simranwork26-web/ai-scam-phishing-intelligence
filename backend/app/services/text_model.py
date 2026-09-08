from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "results/models/distilbert"
MAX_LENGTH = 256


class TextModel:
    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            probability = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

        return float(probability)


_model = None


def get_text_model() -> TextModel:
    global _model
    if _model is None:
        _model = TextModel()
    return _model
