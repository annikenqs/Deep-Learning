import pandas as pd
import random

# ====== PARAMETER ======
SHUFFLE_PERCENTAGE = 0.5  # e.g., 0.1 = 10% of words will be shuffled

# ====== FILES ======
INPUT_FILE = "Dataset/test_data.csv"
OUTPUT_FILE = "Dataset/test_data_shuffle_50.csv"

# ====== LOAD CSV ======
df = pd.read_csv(INPUT_FILE)

BODY_COLUMN = "body"  # change if needed

def shuffle_percentage_words(text, percentage):
    words = str(text).split()
    n = len(words)

    if n < 2:
        return text

    # Number of words to affect
    k = max(1, int(n * percentage))

    # Select k unique indices
    indices = random.sample(range(n), k)

    # Extract selected words
    selected_words = [words[i] for i in indices]

    # Shuffle only those selected words
    random.shuffle(selected_words)

    # Put them back into their original positions
    for idx, word_idx in enumerate(indices):
        words[word_idx] = selected_words[idx]

    return " ".join(words)

# ====== APPLY PER ARTICLE ======
df[BODY_COLUMN] = df[BODY_COLUMN].apply(
    lambda body: shuffle_percentage_words(body, SHUFFLE_PERCENTAGE)
)

# ====== SAVE OUTPUT ======
df.to_csv(OUTPUT_FILE, index=False)

print(f"Done. {SHUFFLE_PERCENTAGE*100}% of words shuffled per article.")