"""Minimal Gradio app for user-facing SketchCritic random forest predictions."""

from __future__ import annotations

from typing import Any

import gradio as gr

from predict import predict_issue

FACE_MODEL_PATH = "face_landmarker.task"
CLASSIFIER_MODEL_PATH = "models/sketchcritic_rf.pkl"
DEFAULT_THRESHOLD = 0.1


def _display_label(raw_label: str) -> str:
    """Remove the label_ prefix for user-facing output."""
    return raw_label.removeprefix("label_")


def _top_confidences(confidences: dict[str, float], limit: int = 5) -> dict[str, float]:
    """Return the highest-confidence labels for compact display."""
    sorted_items = sorted(confidences.items(), key=lambda item: item[1], reverse=True)
    return {
        _display_label(label): score
        for label, score in sorted_items[:limit]
    }


def run_app_prediction(
    image: Any,
) -> tuple[str, str, dict[str, float], dict[str, float] | None]:
    """Run random forest prediction for one uploaded image and hide rule-based details."""
    if image is None:
        return ("error: Please upload an image.", "", {}, None)

    result = predict_issue(
        image_path=str(image),
        face_model_path=FACE_MODEL_PATH,
        classifier_model_path=CLASSIFIER_MODEL_PATH,
        threshold=DEFAULT_THRESHOLD,
    )

    if result["status"] != "ok":
        return (f"{result['status']}: {result['message']}", "", {}, result["features"])

    labels_text = ", ".join(_display_label(label) for label in result["mlp_predicted_labels"])
    return (
        f"{result['status']}: {result['message']}",
        labels_text,
        _top_confidences(result["mlp_confidences"]),
        result["features"],
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
            gr.Textbox(label="Random Forest Predicted Labels"),
            gr.JSON(label="Top Random Forest Confidence Scores"),
            gr.JSON(label="Feature Dictionary"),
        ],
        title="SketchCritic",
        description="Upload a face image to view random-forest-based proportion feedback.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
