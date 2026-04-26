"""MediaPipe Face Landmarker extraction for one image."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

# Configure the MediaPipe face landmarker used throughout the pipeline.
def create_face_landmarker(model_path: str) -> vision.FaceLandmarker:
    """Create a MediaPipe Face Landmarker for single-image inference."""
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)

# Load an input image and force it into RGB before passing it to MediaPipe.
def _load_mp_image(image_path: str) -> mp.Image:
    """Load an image and normalize it to RGB for MediaPipe."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        with NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = handle.name
        try:
            rgb_image.save(temp_path)
            return mp.Image.create_from_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

# Extract exactly one face worth of landmarks and package them into a simple dict.
def extract_landmarks(image_path: str, model_path: str) -> dict[str, Any]:
    """Load an image, detect exactly one face, and return structured landmarks."""
    image_file = Path(image_path)
    model_file = Path(model_path)

    if not image_file.is_file():
        raise FileNotFoundError(f"Image file not found: {image_file}")
    if not model_file.is_file():
        raise FileNotFoundError(
            "MediaPipe face landmarker model not found. "
            f"Expected a .task file at: {model_file}"
        )

    mp_image = _load_mp_image(str(image_file))

    with create_face_landmarker(str(model_file)) as landmarker:
        result = landmarker.detect(mp_image)

    face_count = len(result.face_landmarks)
    if face_count == 0:
        raise ValueError("No face detected in the image.")
    if face_count > 1:
        raise ValueError(f"Expected exactly one face, but found {face_count}.")

    landmarks = result.face_landmarks[0]
    structured_points = [
        {"index": index, "x": point.x, "y": point.y, "z": point.z}
        for index, point in enumerate(landmarks)
    ]

    return {
        "status": "ok",
        "image_path": str(image_file),
        "face_count": face_count,
        "landmark_count": len(structured_points),
        "landmarks": structured_points,
    }

# Build the small CLI parser for standalone landmark testing.
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Face Landmarker on a single image."
    )
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL"),
        help=(
            "Path to the MediaPipe face landmarker .task model file. "
            "Defaults to MEDIAPIPE_FACE_LANDMARKER_MODEL."
        ),
    )
    return parser

# Run the standalone landmark extraction command-line entry point.
def main() -> None:
    """Simple CLI entry point for quick landmark extraction checks."""
    parser = _build_parser()
    args = parser.parse_args()

    if not args.model_path:
        parser.error(
            "A MediaPipe .task model file is required. "
            "Pass --model-path or set MEDIAPIPE_FACE_LANDMARKER_MODEL."
        )

    result = extract_landmarks(args.image_path, args.model_path)

    print(f"Status: {result['status']}")
    print(f"Faces detected: {result['face_count']}")
    print(f"Landmark count: {result['landmark_count']}")
    print("First five landmarks:")
    for point in result["landmarks"][:5]:
        print(
            f"  {point['index']}: "
            f"x={point['x']:.4f}, y={point['y']:.4f}, z={point['z']:.4f}"
        )


if __name__ == "__main__":
    main()
