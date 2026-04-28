import pandas as pd


DATASET_PATH = "Dataset/training_data.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    # Clean body text
    df["body"] = df["body"].fillna("").astype(str)

    # Word count
    df["body_length_words"] = df["body"].apply(lambda x: len(x.split()))

    # Character count (optional but useful)
    df["body_length_chars"] = df["body"].apply(len)

    print("\n=== Article length (words) ===")
    print(df.groupby("label")["body_length_words"].describe())

    print("\n=== Average article length (words) ===")
    print(df.groupby("label")["body_length_words"].mean())

    print("\n=== Article length (characters) ===")
    print(df.groupby("label")["body_length_chars"].mean())

    # Short vs long articles (based on median)
    median_length = df["body_length_words"].median()
    df["long_article"] = df["body_length_words"] > median_length

    print("\n=== Long vs short articles ===")
    print(pd.crosstab(df["label"], df["long_article"], normalize="index"))

    print("\n=== Raw counts ===")
    print(pd.crosstab(df["label"], df["long_article"]))


if __name__ == "__main__":
    main()