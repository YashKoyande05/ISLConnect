"""
Train the Phase 1 ISL classifier.

Input:
    dataset/processed/landmarks.csv

Output:
    model/isl_model.joblib
    model/labels.json
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


CSV_PATH = Path("dataset/processed/landmarks.csv")
MODEL_DIR = Path("model")


def main():
    if not CSV_PATH.exists():
        raise SystemExit(
            "Landmark CSV not found.\n"
            "Run first: python -m training.extract_landmarks"
        )

    df = pd.read_csv(CSV_PATH)

    feat_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    feat_cols.sort(key=lambda c: int(c[1:]))
    print(f"Found {len(feat_cols)} landmark features per sample.")

    X = df[feat_cols].values.astype(np.float32)
    y_text = df["label"].astype(str).values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    classes = list(encoder.classes_)

    print("Classes:", classes)
    print("Number of classes:", len(classes))
    print("Samples:", len(X))

    # Stratified split keeps every class represented in train/test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,  # Stronger L2 regularization to prevent overfitting
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=True,
    )

    print("\nTraining...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=classes,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "isl_model.joblib")

    (MODEL_DIR / "labels.json").write_text(
        json.dumps(classes, indent=2),
        encoding="utf-8"
    )

    print("\nSaved:")
    print(MODEL_DIR / "isl_model.joblib")
    print(MODEL_DIR / "labels.json")


if __name__ == "__main__":
    main()
