from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

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

from dataset import load_splits


DISTILBERT_MODEL_DIR = "models/distilbert_default"
MAX_LENGTH = 256


def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Real", "Actual Fake"],
        columns=["Pred Real", "Pred Fake"],
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["real", "fake"], digits=4))


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["input_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

def run_distilbert_base_inference(test_df):
    print("\n--- DistilBERT (Base, no fine-tuning) ---")

    test_hf = Dataset.from_pandas(
        test_df[["input_text", "label_id"]].rename(columns={"label_id": "labels"})
    )

    model_name = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    test_hf = test_hf.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    test_hf = test_hf.remove_columns(["input_text"])

    if "__index_level_0__" in test_hf.column_names:
        test_hf = test_hf.remove_columns(["__index_level_0__"])

    test_hf.set_format("torch")

    dataloader = DataLoader(test_hf, batch_size=32)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            print("Processing batch...")
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = test_df["label_id"].to_numpy()

    evaluate("DistilBERT Base (Test)", y_true, y_pred)


def run_baseline_inference(test_df):
    logreg_model_path = Path("models") / "tfidf_logreg.joblib"
    nb_model_path = Path("models") / "tfidf_naive_bayes.joblib"

    X_test = test_df["input_text"]
    y_test = test_df["label_id"]

    if nb_model_path.exists():
        print("\n--- Naive Bayes + TF-IDF ---")
        nb_model = joblib.load(nb_model_path)
        y_pred_nb = nb_model.predict(X_test)
        evaluate("Naive Bayes (Test)", y_test, y_pred_nb)
    else:
        print("\nNaive Bayes model not found, skipping.")

    if logreg_model_path.exists():
        print("\n--- Logistic Regression + TF-IDF ---")
        logreg_model = joblib.load(logreg_model_path)
        y_pred_logreg = logreg_model.predict(X_test)
        evaluate("Logistic Regression (Test)", y_test, y_pred_logreg)
    else:
        print("\nLogistic Regression model not found, skipping.")


def run_distilbert_inference(test_df):
    model_path = Path(DISTILBERT_MODEL_DIR)

    if not model_path.exists():
        print("\nDistilBERT model not found, skipping.")
        return

    test_hf = Dataset.from_pandas(
        test_df[["input_text", "label_id"]].rename(columns={"label_id": "labels"})
    )

    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)

    test_hf = test_hf.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    test_hf = test_hf.remove_columns(["input_text"])

    if "__index_level_0__" in test_hf.column_names:
        test_hf = test_hf.remove_columns(["__index_level_0__"])

    test_hf.set_format("torch")

    trainer = Trainer(model=model)

    predictions = trainer.predict(test_hf)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_df["label_id"].to_numpy()

    print("\n--- DistilBERT ---")
    evaluate("DistilBERT (Test)", y_true, y_pred)


def main():
    _, _, test_df = load_splits("Dataset")

    print("\n=== Running inference on test set ===")

    run_baseline_inference(test_df)
    run_distilbert_inference(test_df)
    run_distilbert_base_inference(test_df)


if __name__ == "__main__":
    main()