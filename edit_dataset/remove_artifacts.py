from pathlib import Path
import pandas as pd
import re

DATASET_DIR = Path("Dataset")


def clean_artifacts(text):
    text = str(text)

    # Remove source marker at beginning, e.g. "PARIS (Reuters) -"
    text = re.sub(r"^[A-Z\s,'\.-]+\(Reuters\)\s*[-–—]\s*", "", text)

    # Remove "(Reuters)" anywhere
    text = re.sub(r"\(Reuters\)", "", text, flags=re.IGNORECASE)

    # Remove the word "Reuters" anywhere
    text = re.sub(r"\bReuters\b", "", text, flags=re.IGNORECASE)

    # Remove bracketed video markers: (VIDEO), [Video], etc.
    text = re.sub(r"[\(\[]\s*videos?\s*[\)\]]", "", text, flags=re.IGNORECASE)

    # Clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def process_file(file_path):
    df = pd.read_csv(file_path)

    df["title"] = df["title"].fillna("").apply(clean_artifacts)
    df["body"] = df["body"].fillna("").apply(clean_artifacts)

    # Overwrite the original file
    df.to_csv(file_path, index=False)
    print(f"Overwritten: {file_path}")


def main():
    process_file(DATASET_DIR / "training_data.csv")
    process_file(DATASET_DIR / "validation_data.csv")
    process_file(DATASET_DIR / "test_data.csv")


if __name__ == "__main__":
    main()