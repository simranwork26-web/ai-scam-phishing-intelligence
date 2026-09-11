from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/text_splits"
OUTPUT_DIR = ROOT / "results/models/distilbert_short_benign"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
SEED = 42


def load_split(filename: str) -> Dataset:
    df = pd.read_csv(DATA_DIR / filename)
    return Dataset.from_pandas(
        df[["text", "label"]],
        preserve_index=False,
    )


def main() -> None:
    set_seed(SEED)

    train_dataset = load_split("train.csv")
    benign_df = pd.read_csv(
        ROOT / "data/processed/modern_benign_augmentation.csv"
    )
    malicious_df = pd.read_csv(
        ROOT / "data/processed/modern_malicious_augmentation.csv"
    )
    short_benign_df = pd.read_csv(
        ROOT / "data/processed/short_benign_augmentation.csv"
    )
    benign_dataset = Dataset.from_pandas(
        benign_df[["text", "label"]],
        preserve_index=False,
    )
    malicious_dataset = Dataset.from_pandas(
        malicious_df[["text", "label"]],
        preserve_index=False,
    )
    short_benign_dataset = Dataset.from_pandas(
        short_benign_df[["text", "label"]],
        preserve_index=False,
    )
    train_dataset = Dataset.from_dict({
        "text": (
            list(train_dataset["text"])
            + list(benign_dataset["text"])
            + list(malicious_dataset["text"])
            + list(short_benign_dataset["text"])
        ),
        "label": (
            list(train_dataset["label"])
            + list(benign_dataset["label"])
            + list(malicious_dataset["label"])
            + list(short_benign_dataset["label"])
        ),
    })
    validation_dataset = load_split("validation.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probabilities = torch.softmax(
            torch.tensor(logits),
            dim=-1,
        ).numpy()[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(
                labels, predictions, zero_division=0
            ),
            "recall": recall_score(
                labels, predictions, zero_division=0
            ),
            "f1": f1_score(
                labels, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(
                labels, probabilities
            ),
        }

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("=== DistilBERT Fine-Tuning ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Device: {'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    print()

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print()
    print(f"Best model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
