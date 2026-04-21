import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "models/distilbert_default"
MAX_LENGTH = 256

# Load once so LIME can call predict_proba many times efficiently
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


def predict_proba(texts):
    """
    LIME-compatible prediction function.
    Input: list[str]
    Output: np.ndarray of shape (n_samples, 2)
    """
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
    sample_texts = [
        "The government announced a new policy today.",
        "Breaking!!! You won't believe this shocking secret!!!"
    ]

    probs = predict_proba(sample_texts)
    print(probs)
    print(probs.shape)  # should be (2, 2)