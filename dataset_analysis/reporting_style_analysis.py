import pandas as pd


DATASET_PATH = "Dataset/training_data.csv"

REPORTING_WORDS = [
    "said",
    "told",
    "according",
    "reported",
    "statement",
    "spokesman",
    "spokeswoman",
    "reporters",
]


def count_reporting_words(text):
    text = str(text).lower()
    return sum(text.count(word) for word in REPORTING_WORDS)


def main():
    df = pd.read_csv(DATASET_PATH)

    df["text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    )

    df["reporting_count"] = df["text"].apply(count_reporting_words)

    print("\n=== Reporting word usage ===")
    print(df.groupby("label")["reporting_count"].describe())

    print("\n=== Average reporting words per article ===")
    print(df.groupby("label")["reporting_count"].mean())

    df["has_reporting_words"] = df["reporting_count"] > 0

    print("\n=== Articles containing at least one reporting word ===")
    print(pd.crosstab(df["label"], df["has_reporting_words"], normalize="index"))

    print("\n=== Raw counts ===")
    print(pd.crosstab(df["label"], df["has_reporting_words"]))


if __name__ == "__main__":
    main()