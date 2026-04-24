"""Prediction pipeline for trained SketchCritic model inference."""

from __future__ import annotations

import argparse
import pickle
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd

from features import compute_features
from landmarks import extract_landmarks

BALANCED_LABEL_CANDIDATES = ("label_balanced", "balanced")
BALANCED_MARGIN = 0.01
OPPOSING_LABEL_PAIRS = [
    ("both_eyes_appear_large", "both_eyes_appear_small"),
    ("nose_too_high", "nose_too_low"),
    ("mouth_too_high", "mouth_too_low"),
    ("left_eye_appears_larger", "right_eye_appears_larger"),
    ("left_eye_too_high", "right_eye_too_high"),
    ("left_mouth_corner_too_high", "right_mouth_corner_too_high"),
    (
        "nose_too_narrow_relative_to_inner_eye_gap",
        "nose_too_wide_relative_to_inner_eye_gap",
    ),
]


def _load_model_artifact(model_path: str) -> dict[str, Any]:
    """Load a trained model artifact from disk."""
    with open(model_path, "rb") as handle:
        artifact = pickle.load(handle)
    return artifact


def _default_balanced_label(label_names: list[str]) -> str:
    """Return the stored balanced label name when possible."""
    for candidate in ("label_balanced", "balanced"):
        if candidate in label_names:
            return candidate
    return "balanced"


def _canonical_label(label: str) -> str:
    """Normalize a label for internal comparison."""
    return label.removeprefix("label_")


def _raw_label_map(label_names: list[str]) -> dict[str, str]:
    """Map normalized labels back to the raw labels stored in the artifact."""
    return {_canonical_label(label): label for label in label_names}


def postprocess_mlp_predictions(
    probabilities: dict[str, float], threshold: float
) -> list[str]:
    """Convert per-label probabilities into final multi-label predictions."""
    if not probabilities:
        return ["balanced"]

    raw_map = _raw_label_map(list(probabilities.keys()))
    balanced_raw = _default_balanced_label(list(probabilities.keys()))
    balanced_canonical = _canonical_label(balanced_raw)

    canonical_probs = {
        _canonical_label(label): score for label, score in probabilities.items()
    }
    balanced_score = canonical_probs.get(balanced_canonical, 0.0)

    highest_label = max(canonical_probs, key=canonical_probs.get)
    if highest_label == balanced_canonical:
        return [balanced_raw]

    if canonical_probs[highest_label] < balanced_score + BALANCED_MARGIN:
        return [balanced_raw]

    selected = [
        label
        for label, score in canonical_probs.items()
        if label != balanced_canonical and score >= threshold
    ]

    if not selected:
        return [balanced_raw]

    for left_label, right_label in OPPOSING_LABEL_PAIRS:
        if left_label in selected and right_label in selected:
            if canonical_probs[left_label] >= canonical_probs[right_label]:
                selected.remove(right_label)
            else:
                selected.remove(left_label)

    if not selected:
        return [balanced_raw]

    return [raw_map[label] for label in selected]


def _extract_confidences(
    classifier: Any, feature_vector: Any, label_names: list[str]
) -> dict[str, float]:
    """Return per-label confidence scores when the classifier supports them."""
    if not hasattr(classifier, "predict_proba"):
        return {}

    probabilities = classifier.predict_proba(feature_vector)
    confidences: dict[str, float] = {}

    if isinstance(probabilities, list):
        for label_name, probs in zip(label_names, probabilities, strict=False):
            probs_array = np.asarray(probs)
            if probs_array.ndim == 2 and probs_array.shape[1] >= 2:
                confidences[label_name] = float(probs_array[0, 1])
            elif probs_array.ndim >= 1:
                confidences[label_name] = float(probs_array.ravel()[-1])
    else:
        probs_array = np.asarray(probabilities)
        if probs_array.ndim == 2 and probs_array.shape[0] == 1:
            for index, label_name in enumerate(label_names):
                if index < probs_array.shape[1]:
                    confidences[label_name] = float(probs_array[0, index])

    return confidences


def _display_label(raw_label: str) -> str:
    """Strip the label_ prefix for CLI display."""
    return raw_label.removeprefix("label_")


def _ordered_feature_frame(
    feature_dict: dict[str, float], feature_names: list[str]
) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact model feature order."""
    return pd.DataFrame(
        [[feature_dict[name] for name in feature_names]],
        columns=feature_names,
    )


def predict_issue(
    image_path: str,
    face_model_path: str,
    classifier_model_path: str,
    threshold: float = 0.1,
) -> dict[str, Any]:
    """Extract landmarks and features, then run model predictions."""
    try:
        landmark_result = extract_landmarks(image_path, face_model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return {
            "status": "error",
            "features": None,
            "mlp_predicted_labels": None,
            "mlp_confidences": None,
            "message": f"Landmark extraction failed: {exc}",
        }

    try:
        feature_dict = compute_features(landmark_result)
    except (ValueError, KeyError) as exc:
        return {
            "status": "error",
            "features": None,
            "mlp_predicted_labels": None,
            "mlp_confidences": None,
            "message": f"Feature computation failed: {exc}",
        }

    try:
        artifact = _load_model_artifact(classifier_model_path)
    except (FileNotFoundError, OSError, pickle.PickleError) as exc:
        return {
            "status": "error",
            "features": feature_dict,
            "mlp_predicted_labels": None,
            "mlp_confidences": None,
            "message": f"Model loading failed: {exc}",
        }

    classifier = artifact.get("model")
    feature_names = artifact.get("feature_names")
    label_names = artifact.get("label_names")

    if classifier is None or feature_names is None or label_names is None:
        return {
            "status": "error",
            "features": feature_dict,
            "mlp_predicted_labels": None,
            "mlp_confidences": None,
            "message": (
                "Model artifact is missing one or more required keys: "
                "'model', 'feature_names', 'label_names'."
            ),
        }

    missing_features = [name for name in feature_names if name not in feature_dict]
    if missing_features:
        return {
            "status": "error",
            "features": feature_dict,
            "mlp_predicted_labels": None,
            "mlp_confidences": None,
            "message": (
                "Computed features are missing values required by the model: "
                + ", ".join(missing_features)
            ),
        }

    feature_frame = _ordered_feature_frame(feature_dict, list(feature_names))

    try:
        mlp_confidences = _extract_confidences(
            classifier, feature_frame, list(label_names)
        )
        if mlp_confidences:
            mlp_predicted_labels = postprocess_mlp_predictions(
                mlp_confidences, threshold
            )
        else:
            prediction = classifier.predict(feature_frame)
            binary = np.asarray(prediction).astype(int).ravel()
            mlp_predicted_labels = [
                label
                for label, is_active in zip(label_names, binary, strict=False)
                if is_active == 1
            ]
            if not mlp_predicted_labels:
                mlp_predicted_labels = [_default_balanced_label(list(label_names))]
            mlp_confidences = {}
    except Exception as exc:
        return {
            "status": "error",
            "features": feature_dict,
            "mlp_predicted_labels": None,
            "mlp_confidences": None,
            "message": f"MLP prediction failed: {exc}",
        }

    return {
        "status": "ok",
        "features": feature_dict,
        "mlp_predicted_labels": mlp_predicted_labels,
        "mlp_confidences": mlp_confidences,
        "message": "Model predictions computed successfully.",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SketchCritic model prediction on one face image."
    )
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument("face_model_path", help="Path to the face landmarker .task file.")
    parser.add_argument(
        "classifier_model_path", help="Path to the trained SketchCritic model artifact."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Probability threshold for non-balanced model labels.",
    )
    return parser


def main() -> None:
    """Run the model prediction pipeline."""
    parser = _build_parser()
    args = parser.parse_args()
    result = predict_issue(
        args.image_path,
        args.face_model_path,
        args.classifier_model_path,
        threshold=args.threshold,
    )

    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    if result["status"] != "ok":
        return

    print("Features:")
    pprint(result["features"])
    print("Predicted labels:")
    pprint([_display_label(label) for label in result["mlp_predicted_labels"]])
    print("Confidences:")
    formatted_confidences = {
        _display_label(label): confidence
        for label, confidence in result["mlp_confidences"].items()
    }
    pprint(formatted_confidences)


if __name__ == "__main__":
    main()
