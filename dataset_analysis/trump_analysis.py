import pandas as pd


DATASET_PATH = "Dataset/training_data.csv"

PHRASES = [
    "president donald trump",
    "president trump",
    "trump",
]


def count_phrase(text, phrase):
    text = str(text).lower()
    return text.count(phrase)


def main():
    df = pd.read_csv(DATASET_PATH)

    df["text"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    )

    for phrase in PHRASES:
        count_col = f"count_{phrase.replace(' ', '_')}"
        present_col = f"has_{phrase.replace(' ', '_')}"

        df[count_col] = df["text"].apply(lambda x: count_phrase(x, phrase))
        df[present_col] = df[count_col] > 0

        print(f"\n=== Phrase: '{phrase}' ===")

        print("\nAverage occurrences per article:")
        print(df.groupby("label")[count_col].mean())

        print("\nProportion of articles containing phrase:")
        print(pd.crosstab(df["label"], df[present_col], normalize="index"))

        print("\nRaw counts:")
        print(pd.crosstab(df["label"], df[present_col]))


if __name__ == "__main__":
    main()