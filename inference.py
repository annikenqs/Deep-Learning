from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dataset import load_splits


def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")

    # Metrics
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Real", "Actual Fake"],
        columns=["Pred Real", "Pred Fake"],
    )

    print("\nConfusion Matrix:")
    print(cm_df)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["real", "fake"]))


def main():
    logreg_model_path = Path("models") / "tfidf_logreg.joblib"
    nb_model_path = Path("models") / "tfidf_naive_bayes.joblib"

    if not logreg_model_path.exists():
        raise FileNotFoundError("Logistic Regression model file not found. Run train.py first.")

    if not nb_model_path.exists():
        raise FileNotFoundError("Naive Bayes model file not found. Run train.py first.")

    _, _, test_df = load_splits("Dataset")

    X_test = test_df["input_text"]
    y_test = test_df["label_id"]

    print("\n=== Running inference on test set ===")

    # --- Naive Bayes ---
    print("\n--- Naive Bayes + TF-IDF ---")
    nb_model = joblib.load(nb_model_path)
    y_test_pred_nb = nb_model.predict(X_test)
    evaluate("Naive Bayes (Test)", y_test, y_test_pred_nb)

    # --- Logistic Regression ---
    print("\n--- Logistic Regression + TF-IDF ---")
    logreg_model = joblib.load(logreg_model_path)
    y_test_pred_logreg = logreg_model.predict(X_test)
    evaluate("Logistic Regression (Test)", y_test, y_test_pred_logreg)


if __name__ == "__main__":
    main()