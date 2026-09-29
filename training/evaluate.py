"""
Evaluate an already-trained model against dataset/processed/landmarks.csv.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CSV_PATH = Path("dataset/processed/landmarks.csv")
MODEL_PATH = Path("model/isl_model.joblib")


def main():
    if not CSV_PATH.exists() or not MODEL_PATH.exists():
        raise SystemExit(
            "Train the model first with: python -m training.train"
        )

    df = pd.read_csv(CSV_PATH)
    X = df[[f"f{i}" for i in range(63)]].values
    y = df["label"].astype(str).values

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X)

    print(f"Accuracy on processed dataset: {accuracy_score(y, pred)*100:.2f}%")
    print(classification_report(y, pred, zero_division=0))


if __name__ == "__main__":
    main()
