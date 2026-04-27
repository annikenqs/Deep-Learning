import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

MODEL_DIR = "models/distilbert_default"
MAX_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs.cpu().numpy()


if __name__ == "__main__":
    failed_df = pd.read_csv("distilbert_failed_examples.csv")

    row = failed_df.iloc[0]
    sample_text = row["input_text"]
    true_label = row["true_label"]
    pred_label = row["pred_label"]

    probs = predict_proba([sample_text])[0]

    print("True label:", true_label)
    print("Predicted label:", pred_label)
    print("Probabilities:", probs)

    explainer = LimeTextExplainer(class_names=["real", "fake"])

    explanation = explainer.explain_instance(
        sample_text,
        predict_proba,
        num_features=10,
        num_samples=1000
    )

    print("\nTop features:")
    for feature, weight in explanation.as_list():
        print(f"{feature}: {weight:.4f}")

    explanation.save_to_file("lime_explanation.html")
    print("\nSaved explanation to lime_explanation.html")