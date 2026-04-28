from pathlib import Path
import joblib
import numpy as np


MODEL_PATH = Path("models") / "tfidf_logreg.joblib"


def main():
    model = joblib.load(MODEL_PATH)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["logreg"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = classifier.coef_[0]

    # Since fake = 1 and real = 0:
    # positive coefficients push prediction toward fake
    # negative coefficients push prediction toward real
    top_fake_idx = np.argsort(coefficients)[-30:][::-1]
    top_real_idx = np.argsort(coefficients)[:30]

    print("\n=== Top features associated with FAKE ===")
    for feature, weight in zip(feature_names[top_fake_idx], coefficients[top_fake_idx]):
        print(f"{feature:30s} {weight:.4f}")

    print("\n=== Top features associated with REAL ===")
    for feature, weight in zip(feature_names[top_real_idx], coefficients[top_real_idx]):
        print(f"{feature:30s} {weight:.4f}")


if __name__ == "__main__":
    main()