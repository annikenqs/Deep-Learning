from pathlib import Path
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)

from src.dataset import load_splits


MODEL_DIR = "models/distilbert_default"
MAX_LENGTH = 256


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


def main():
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            f"Train the model first with: python train_distilbert.py"
        )

    _, _, test_df = load_splits("Dataset")

    test_hf = Dataset.from_pandas(
        test_df[["input_text", "label_id"]].rename(columns={"label_id": "labels"})
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    test_hf = test_hf.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    test_hf = test_hf.remove_columns(["input_text"])

    for col in ["__index_level_0__"]:
        if col in test_hf.column_names:
            test_hf = test_hf.remove_columns([col])

    test_hf.set_format("torch")

    trainer = Trainer(
        model=model,
    )

    predictions = trainer.predict(test_hf)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df["label_id"].to_numpy()

    print("\n=== DistilBERT (Test) ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Pred 0", "Pred 1"],
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()