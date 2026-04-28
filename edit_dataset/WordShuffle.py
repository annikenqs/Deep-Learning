import pandas as pd
import random

SHUFFLE_PERCENTAGE = 0.2

INPUT_FILE = "Dataset/test_data.csv"
OUTPUT_FILE = "Dataset/test_data_shuffle_20.csv"

df = pd.read_csv(INPUT_FILE)

def shuffle_percentage_words(text, percentage):
    words = str(text).split()
    n = len(words)

    if n < 2:
        return text

    k = max(1, int(n * percentage))
    indices = random.sample(range(n), k)

    selected_words = [words[i] for i in indices]
    random.shuffle(selected_words)

    for i, word_idx in enumerate(indices):
        words[word_idx] = selected_words[i]

    return " ".join(words)

# Shuffle BOTH title and body
df["title"] = df["title"].fillna("").apply(
    lambda x: shuffle_percentage_words(x, SHUFFLE_PERCENTAGE)
)

df["body"] = df["body"].fillna("").apply(
    lambda x: shuffle_percentage_words(x, SHUFFLE_PERCENTAGE)
)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Done. {SHUFFLE_PERCENTAGE*100}% of words shuffled in title and body.")