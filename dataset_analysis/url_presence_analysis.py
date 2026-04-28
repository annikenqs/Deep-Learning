import pandas as pd


DATASET_PATH = "Dataset/training_data.csv"


def has_url(text):
    text = str(text).lower()
    return (
        "http://" in text
        or "https://" in text
        or "www." in text
        or "twitter.com" in text
        or "facebook.com" in text
    )


def main():
    df = pd.read_csv(DATASET_PATH)

    df["text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    )

    df["has_url"] = df["text"].apply(has_url)

    print("\n=== URL presence (proportion) ===")
    print(pd.crosstab(df["label"], df["has_url"], normalize="index"))

    print("\n=== URL presence (counts) ===")
    print(pd.crosstab(df["label"], df["has_url"]))


if __name__ == "__main__":
    main()