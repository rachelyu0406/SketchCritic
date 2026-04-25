"""Train a small multi-label MLP on synthetic SketchCritic data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Load the CSV and split it into feature and label tables."""
    data = pd.read_csv(csv_path)

    label_names = [column for column in data.columns if column.startswith("label_")]
    if not label_names:
        raise ValueError("No label columns found. Expected columns starting with 'label_'.")

    # Keep numeric non-label columns as features and ignore summary text columns
    # such as 'active_labels'.
    feature_names = [
        column
        for column in data.columns
        if not column.startswith("label_") and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not feature_names:
        raise ValueError("No numeric feature columns found in the dataset.")

    X = data[feature_names]
    y = data[label_names]
    return X, y, feature_names, label_names


def create_model(random_state: int = 42) -> Pipeline:
    """Create a small MLP pipeline for multi-label classification."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(48, 24),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.0005,
                    alpha=0.0005,
                    batch_size=64,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    max_iter=1500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_evaluate(
    csv_path: str,
    model_output_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train the model, print evaluation metrics, and save the artifact."""
    X, y, feature_names, label_names = load_dataset(csv_path)
    print(f"Loaded {len(feature_names)} feature columns and {len(label_names)} label columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = create_model(random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    exact_match_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred, average="micro", zero_division=0)
    overall_recall = recall_score(y_test, y_pred, average="micro", zero_division=0)
    overall_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    report_df = pd.DataFrame(
        classification_report(
            y_test,
            y_pred,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
    ).T.drop(columns=["support"], errors="ignore")
    report = report_df.to_string()

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "label_names": label_names,
    }

    output_file = Path(model_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as handle:
        pickle.dump(artifact, handle)

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "classification_report": report,
        "model_path": str(output_file),
        "feature_names": feature_names,
        "label_names": label_names,
    }


def main() -> None:
    """Command-line entry point for training the multi-label model."""
    parser = argparse.ArgumentParser(
        description="Train a multi-label SketchCritic MLP on synthetic CSV data."
    )
    parser.add_argument("csv_path", help="Path to the synthetic CSV dataset.")
    parser.add_argument("model_output_path", help="Path to save the trained model artifact.")
    args = parser.parse_args()

    results = train_and_evaluate(args.csv_path, args.model_output_path)

    print(f"Overall accuracy: {results['exact_match_accuracy']:.4f}")
    print(f"Overall precision: {results['overall_precision']:.4f}")
    print(f"Overall recall: {results['overall_recall']:.4f}")
    print(f"Overall F1: {results['overall_f1']:.4f}")
    print("Classification report:")
    print(results["classification_report"])
    print(f"Saved model artifact to: {results['model_path']}")


if __name__ == "__main__":
    main()
