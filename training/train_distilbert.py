from pathlib import Path
import numpy as np

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from dataset import load_splits


MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert_default"
MAX_LENGTH = 256


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def main():
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _ = load_splits("Dataset")

    train_hf = Dataset.from_pandas(
        train_df[["input_text", "label_id"]].rename(columns={"label_id": "labels"})
    )
    val_hf = Dataset.from_pandas(
        val_df[["input_text", "label_id"]].rename(columns={"label_id": "labels"})
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_hf = train_hf.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )
    val_hf = val_hf.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    train_hf = train_hf.remove_columns(["input_text"])
    val_hf = val_hf.remove_columns(["input_text"])

    # Remove pandas index column if it appears
    for col in ["__index_level_0__"]:
        if col in train_hf.column_names:
            train_hf = train_hf.remove_columns([col])
        if col in val_hf.column_names:
            val_hf = val_hf.remove_columns([col])

    train_hf.set_format("torch")
    val_hf.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    results = trainer.evaluate()
    print("\n=== DistilBERT Validation Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()