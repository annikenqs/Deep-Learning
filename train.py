from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from dataset import load_splits
from src.logisticReg_TF_IDF import create_logistic_reg_tfidf_model
from src.naive_bayes_TF_IDF import create_naive_bayes_tfidf_model


def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")


def main():
    train_df, val_df, _ = load_splits("Dataset")

    X_train = train_df["input_text"]
    y_train = train_df["label_id"]

    X_val = val_df["input_text"]
    y_val = val_df["label_id"]

    # --- Naive Bayes ---
    print("\n--- Naive Bayes + TF-IDF ---")

    nb_model = create_naive_bayes_tfidf_model()
    nb_model.fit(X_train, y_train)

    y_val_pred_nb = nb_model.predict(X_val)
    evaluate("Naive Bayes (Validation)", y_val, y_val_pred_nb)

    nb_path = Path("models") / "tfidf_naive_bayes.joblib"
    joblib.dump(nb_model, nb_path)

    # --- Logistic Regression ---
    print("\n--- Logistic Regression + TF-IDF ---")

    model = create_logistic_reg_tfidf_model()
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    evaluate("Logistic Regression (Validation)", y_val, y_val_pred)

    save_path = Path("models") / "tfidf_logreg.joblib"
    joblib.dump(model, save_path)


if __name__ == "__main__":
    main()