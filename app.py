"""Minimal Gradio app for user-facing SketchCritic random forest predictions."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Any

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_issue
from train_random_forest import train_and_evaluate

FACE_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
CLASSIFIER_MODEL_PATH = PROJECT_ROOT / "models" / "sketchcritic_rf.pkl"
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "data" / "sketchcritic_synthetic.csv"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_THRESHOLD = 0.1


def ensure_face_model() -> None:
    """Download the MediaPipe face model if it is not present locally."""
    if FACE_MODEL_PATH.is_file():
        return

    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)


def ensure_classifier_model() -> None:
    """Train the random forest if the saved artifact is missing."""
    if CLASSIFIER_MODEL_PATH.is_file():
        return
    if not SYNTHETIC_DATA_PATH.is_file():
        raise FileNotFoundError(
            "Synthetic dataset not found for auto-training. "
            f"Expected: {SYNTHETIC_DATA_PATH}"
        )

    train_and_evaluate(
        csv_path=str(SYNTHETIC_DATA_PATH),
        model_output_path=str(CLASSIFIER_MODEL_PATH),
    )


def ensure_runtime_assets() -> None:
    """Ensure required runtime assets exist for the Space."""
    ensure_face_model()
    ensure_classifier_model()


def _display_label(raw_label: str) -> str:
    """Remove the label_ prefix for user-facing output."""
    return raw_label.removeprefix("label_")


def _format_detected_issues(labels: list[str]) -> str:
    """Render model labels as a simple Markdown bullet list."""
    display_labels = [_display_label(label) for label in labels]
    if not display_labels:
        return "Detected proportion issues:\n- balanced"
    return "Detected proportion issues:\n" + "\n".join(
        f"- {label}" for label in display_labels
    )


def run_app_prediction(
    image: Any,
) -> tuple[str, str]:
    """Run random forest prediction for one uploaded image and hide rule-based details."""
    if image is None:
        return ("error: Please upload an image.", "Detected proportion issues:")

    try:
        ensure_runtime_assets()
    except Exception as exc:
        return (f"error: Failed to prepare model assets: {exc}", "Detected proportion issues:")

    result = predict_issue(
        image_path=str(image),
        face_model_path=str(FACE_MODEL_PATH),
        classifier_model_path=str(CLASSIFIER_MODEL_PATH),
        threshold=DEFAULT_THRESHOLD,
    )

    if result["status"] != "ok":
        return (
            f"{result['status']}: {result['message']}",
            "Detected proportion issues:",
        )

    return (
        f"{result['status']}: {result['message']}",
        _format_detected_issues(result["mlp_predicted_labels"]),
    )


def main() -> None:
    """Launch the minimal Gradio interface."""
    demo = gr.Interface(
        fn=run_app_prediction,
        inputs=[
            gr.Image(type="filepath", label="Upload Image"),
        ],
        outputs=[
            gr.Textbox(label="Status / Message"),
            gr.Markdown(label="Detected proportion issues:"),
        ],
        title="SketchCritic",
        description="Upload a face image to view random-forest-based proportion feedback.",
        flagging_mode="never",
    )
    demo.launch()


if __name__ == "__main__":
    main()
