from pathlib import Path
import pandas as pd

# Mapping from string labels to numeric labels
LABEL_MAP = {
    "real": 0,
    "fake": 1,
}


def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Check required columns exist
    required_cols = {"title", "body", "date", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    df = df.copy()

    # Fill missing text fields
    df["title"] = df["title"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # Normalize labels (lowercase, strip spaces)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Keep only valid labels
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()

    # Combine title + body into one input text
    df["input_text"] = (
        df["title"] + " " + df["body"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # Remove empty rows
    df = df[df["input_text"].str.len() > 0].copy()

    # Convert labels to numeric
    df["label_id"] = df["label"].map(LABEL_MAP)

    return df


def load_splits(dataset_dir="Dataset"):
    dataset_dir = Path(dataset_dir)

    # Load train/val/test splits
    train_df = load_csv(dataset_dir / "training_data.csv")
    val_df = load_csv(dataset_dir / "validation_data.csv")
    test_df = load_csv(dataset_dir / "test_data.csv")

    return train_df, val_df, test_df
