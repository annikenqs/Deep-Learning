import pandas as pd

def main():
    # Load dataset
    df = pd.read_csv("Dataset/training_data.csv")

    # Combine text
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")

    # Count exclamation marks
    df["exclamation_count"] = df["text"].str.count("!")

    # Also normalize by length (important!)
    df["word_count"] = df["text"].str.split().apply(len)
    df["exclamation_ratio"] = df["exclamation_count"] / df["word_count"]

    # Results
    print("\n=== Exclamation count (raw) ===")
    print(df.groupby("label")["exclamation_count"].describe())

    print("\n=== Exclamation ratio (normalized) ===")
    print(df.groupby("label")["exclamation_ratio"].describe())

    # Binary feature (does it contain any?)
    df["has_exclamation"] = df["exclamation_count"] > 0

    print("\n=== Contains exclamation ===")
    print(pd.crosstab(df["label"], df["has_exclamation"], normalize="index"))
    print(pd.crosstab(df["label"], df["has_exclamation"]))


if __name__ == "__main__":
    main()