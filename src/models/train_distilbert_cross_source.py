from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/text_splits"
MODEL_BASE = "distilbert-base-uncased"

MAX_LENGTH = 256
SEED = 42


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        ),
        batched=True,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.softmax(
        torch.tensor(logits), dim=-1
    ).numpy()[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def train_for_source(source):
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    validation_df = pd.read_csv(DATA_DIR / "validation.csv")

    train_df = train_df[train_df["source"] == source].copy()
    validation_df = validation_df[validation_df["source"] == source].copy()

    print(f"\n=== Training DistilBERT on {source} ===")
    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Training labels:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Validation labels:\n{validation_df['label'].value_counts().sort_index()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]],
        preserve_index=False,
    )
    validation_dataset = Dataset.from_pandas(
        validation_df[["text", "label"]],
        preserve_index=False,
    )

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    validation_dataset = tokenize_dataset(validation_dataset, tokenizer)

    train_dataset = train_dataset.remove_columns(["text"])
    validation_dataset = validation_dataset.remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_BASE,
        num_labels=2,
    )

    output_dir = ROOT / "results/models" / f"distilbert_cross_source_{source.lower()}"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSaved model to: {output_dir}")


def main():
    train_for_source("Ling")
    train_for_source("SpamAssasin")


if __name__ == "__main__":
    main()
