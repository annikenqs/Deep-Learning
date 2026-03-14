import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score

# -------------------------
# 1. Load dataset
# -------------------------

# Path to your CSV file
csv_file = "../Dataset/test_data.csv"

df = pd.read_csv(csv_file, nrows=500)

# -------------------------
# 2. Prepare text input
# -------------------------

# Combine title and body into a single text input
df["text"] = df["title"].astype(str) + " " + df["body"].astype(str)

texts = df["text"].tolist()

# -------------------------
# 3. Load model
# -------------------------

classifier = pipeline(
    "text-classification",
    model="fake-news-bert",
    tokenizer="dhruvpal/fake-news-bert",
    truncation=True
)

# -------------------------
# 4. Run predictions
# -------------------------

results = classifier(texts, batch_size=16)

# Convert model outputs to labels
predictions = []

for r in results:
    label = r["label"]

    # Normalize label format
    if label.lower() in ["fake", "label_1"]:
        predictions.append("fake")
    else:
        predictions.append("real")

df["prediction"] = predictions

# -------------------------
# 5. Normalize true labels
# -------------------------

df["label"] = df["label"].str.lower()

# -------------------------
# 6. Compute accuracy
# -------------------------

accuracy = accuracy_score(df["label"], df["prediction"])

print("Accuracy:", accuracy)

# -------------------------
# 7. Save results
# -------------------------

df.to_csv("predicted_results.csv", index=False)

print("Predictions saved to predicted_results.csv")