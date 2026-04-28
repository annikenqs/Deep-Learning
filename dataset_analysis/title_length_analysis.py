import pandas as pd


DATASET_PATH = "Dataset/training_data.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    # Clean titles
    df["title"] = df["title"].fillna("").astype(str)

    # Compute title length (number of words)
    df["title_length"] = df["title"].apply(lambda x: len(x.split()))

    print("\n=== Title length statistics ===")
    print(df.groupby("label")["title_length"].describe())

    print("\n=== Average title length ===")
    print(df.groupby("label")["title_length"].mean())

    # Optional: short vs long titles
    df["long_title"] = df["title_length"] > df["title_length"].median()

    print("\n=== Long vs short titles ===")
    print(pd.crosstab(df["label"], df["long_title"], normalize="index"))

    print("\n=== Raw counts ===")
    print(pd.crosstab(df["label"], df["long_title"]))


if __name__ == "__main__":
    main()