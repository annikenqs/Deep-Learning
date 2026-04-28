from pathlib import Path
import joblib
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset import load_splits


def evaluate_group(df, pred_col, model_name):
    print(f"\n=== {model_name} ===")

    for group in ["negative", "neutral", "positive"]:
        subset = df[df["tone_group"] == group]

        if len(subset) == 0:
            continue

        y_true = subset["label_id"]
        y_pred = subset[pred_col]

        # Real articles predicted as fake
        real_subset = subset[subset["label_id"] == 0]
        false_positive_rate = (real_subset[pred_col] == 1).mean() if len(real_subset) > 0 else 0

        print(f"\nTone group: {group}")
        print(f"Samples: {len(subset)}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1-score: {f1_score(y_true, y_pred):.4f}")
        print(f"False positive rate on real articles: {false_positive_rate:.4f}")


def main():
    _, _, test_df = load_splits("Dataset")

    analyzer = SentimentIntensityAnalyzer()

    # Compute sentiment/tone score
    test_df["sentiment_score"] = test_df["input_text"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )

    # Group articles by tone
    test_df["tone_group"] = pd.cut(
        test_df["sentiment_score"],
        bins=[-1, -0.3, 0.3, 1],
        labels=["negative", "neutral", "positive"],
        include_lowest=True,
    )

    print("\nTone group distribution:")
    print(test_df["tone_group"].value_counts())

    print("\nClass distribution per tone group:")
    print(pd.crosstab(test_df["tone_group"], test_df["label"], normalize="index"))

    X_test = test_df["input_text"]

    # Load models
    nb_model = joblib.load(Path("models") / "tfidf_naive_bayes.joblib")
    logreg_model = joblib.load(Path("models") / "tfidf_logreg.joblib")

    # Predict
    test_df["pred_nb"] = nb_model.predict(X_test)
    test_df["pred_logreg"] = logreg_model.predict(X_test)

    # Evaluate by tone group
    evaluate_group(test_df, "pred_nb", "Naive Bayes")
    evaluate_group(test_df, "pred_logreg", "Logistic Regression")


if __name__ == "__main__":
    main()