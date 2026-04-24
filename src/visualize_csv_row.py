"""Visualize one synthetic CSV row as a simple schematic face."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 1000
FACE_WIDTH = 360
FACE_HEIGHT = 560


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a float into a closed interval."""
    return max(low, min(high, value))


def load_row(csv_path: str, row_index: int) -> dict[str, Any]:
    """Load one row from the synthetic CSV by index."""
    data = pd.read_csv(csv_path)
    if row_index < 0 or row_index >= len(data):
        raise ValueError(f"Row index {row_index} is out of range for {len(data)} rows.")
    return data.iloc[row_index].to_dict()


def _face_geometry() -> dict[str, float]:
    """Return reusable face layout constants."""
    face_left = (CANVAS_WIDTH - FACE_WIDTH) / 2
    face_top = 170
    return {
        "face_left": face_left,
        "face_top": face_top,
        "face_right": face_left + FACE_WIDTH,
        "face_bottom": face_top + FACE_HEIGHT,
        "face_center_x": face_left + FACE_WIDTH / 2,
        "face_center_y": face_top + FACE_HEIGHT / 2,
    }


def draw_face(row: dict[str, Any], output_path: str) -> Path:
    """Draw a simple face diagram for one synthetic row and save it as a PNG."""
    geom = _face_geometry()
    image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    face_left = geom["face_left"]
    face_top = geom["face_top"]
    face_right = geom["face_right"]
    face_bottom = geom["face_bottom"]
    face_center_x = geom["face_center_x"]

    # Head oval.
    draw.ellipse((face_left, face_top, face_right, face_bottom), outline="black", width=4)

    face_width = FACE_WIDTH
    face_height = FACE_HEIGHT

    # Eye widths/heights.
    total_eye_width = clamp(
        float(row["both_eyes_width_to_face_width_ratio"]) * face_width,
        80.0,
        220.0,
    )
    left_to_right_width = clamp(float(row["left_eye_width_to_right_eye_width_ratio"]), 0.5, 1.8)
    left_eye_width = total_eye_width * (left_to_right_width / (1.0 + left_to_right_width))
    right_eye_width = total_eye_width - left_eye_width

    left_to_right_height = clamp(
        float(row["left_eye_height_to_right_eye_height_ratio"]), 0.5, 1.8
    )
    base_eye_height = 34.0
    left_eye_height = base_eye_height * left_to_right_height
    right_eye_height = base_eye_height

    nose_width = clamp(float(row["nose_width_to_face_width_ratio"]) * face_width, 24.0, 120.0)
    inner_gap_ratio = max(float(row["nose_width_to_inner_eye_gap_ratio"]), 0.1)
    inner_eye_gap = clamp(nose_width / inner_gap_ratio, 24.0, 110.0)
    eye_spacing = inner_eye_gap + left_eye_width / 2 + right_eye_width / 2

    eye_base_y = face_top + 175
    eye_vertical_delta = float(row["left_eye_center_y_minus_right_eye_center_y_ratio"]) * face_height
    left_eye_center_y = eye_base_y + eye_vertical_delta / 2
    right_eye_center_y = eye_base_y - eye_vertical_delta / 2
    left_eye_center_x = face_center_x - eye_spacing / 2
    right_eye_center_x = face_center_x + eye_spacing / 2

    # Brows as rectangles.
    brow_gap = 22
    brow_height = 10
    brow_width_scale = 1.2
    left_brow_width = left_eye_width * brow_width_scale
    right_brow_width = right_eye_width * brow_width_scale

    draw.rectangle(
        (
            left_eye_center_x - left_brow_width / 2,
            left_eye_center_y - left_eye_height / 2 - brow_gap,
            left_eye_center_x + left_brow_width / 2,
            left_eye_center_y - left_eye_height / 2 - brow_gap + brow_height,
        ),
        outline="black",
        width=3,
    )
    draw.rectangle(
        (
            right_eye_center_x - right_brow_width / 2,
            right_eye_center_y - right_eye_height / 2 - brow_gap,
            right_eye_center_x + right_brow_width / 2,
            right_eye_center_y - right_eye_height / 2 - brow_gap + brow_height,
        ),
        outline="black",
        width=3,
    )

    # Eyes as ovals.
    draw.ellipse(
        (
            left_eye_center_x - left_eye_width / 2,
            left_eye_center_y - left_eye_height / 2,
            left_eye_center_x + left_eye_width / 2,
            left_eye_center_y + left_eye_height / 2,
        ),
        outline="black",
        width=3,
    )
    draw.ellipse(
        (
            right_eye_center_x - right_eye_width / 2,
            right_eye_center_y - right_eye_height / 2,
            right_eye_center_x + right_eye_width / 2,
            right_eye_center_y + right_eye_height / 2,
        ),
        outline="black",
        width=3,
    )

    # Nose as a triangle. Horizontal offset ratios in the CSV are absolute,
    # so this uses a consistent rightward shift for visualization only.
    nose_center_x = face_center_x + float(row["nose_center_offset_ratio"]) * face_width
    nose_center_y = face_top + float(row["nose_vertical_position_ratio"]) * face_height
    nose_height = 82
    draw.polygon(
        [
            (nose_center_x, nose_center_y - nose_height / 2),
            (nose_center_x - nose_width / 2, nose_center_y + nose_height / 2),
            (nose_center_x + nose_width / 2, nose_center_y + nose_height / 2),
        ],
        outline="black",
        width=3,
    )

    # Mouth as an oval.
    mouth_width = clamp(float(row["mouth_width_to_face_width_ratio"]) * face_width, 70.0, 220.0)
    mouth_height = 28
    mouth_center_x = face_center_x + float(row["mouth_center_offset_ratio"]) * face_width
    mouth_center_y = face_top + float(row["mouth_vertical_position_ratio"]) * face_height
    mouth_corner_delta = float(row["left_mouth_corner_y_minus_right_mouth_corner_y_ratio"]) * face_height
    mouth_top = mouth_center_y - mouth_height / 2 + mouth_corner_delta / 4
    mouth_bottom = mouth_center_y + mouth_height / 2 - mouth_corner_delta / 4
    draw.ellipse(
        (
            mouth_center_x - mouth_width / 2,
            mouth_top,
            mouth_center_x + mouth_width / 2,
            mouth_bottom,
        ),
        outline="black",
        width=3,
    )

    # Minimal annotation text.
    active_labels = str(row.get("active_labels", ""))
    draw.text((40, 35), f"row labels: {active_labels}", fill="black")
    draw.text((40, 70), f"middle/lower ratio: {row['middle_lower_max_to_min']:.3f}", fill="black")
    draw.text((40, 95), f"nose vertical ratio: {row['nose_vertical_position_ratio']:.3f}", fill="black")
    draw.text((40, 120), f"mouth vertical ratio: {row['mouth_vertical_position_ratio']:.3f}", fill="black")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_file)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Visualize one SketchCritic synthetic CSV row as a simple face."
    )
    parser.add_argument("csv_path", help="Path to the synthetic CSV file.")
    parser.add_argument("--row", type=int, default=0, help="Zero-based row index to draw.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional PNG output path. Defaults to face_row_<index>.png next to the CSV.",
    )
    return parser


def main() -> None:
    """Load one row and save a face diagram PNG."""
    parser = build_parser()
    args = parser.parse_args()

    row = load_row(args.csv_path, args.row)
    if args.output:
        output_path = args.output
    else:
        csv_file = Path(args.csv_path)
        output_path = str(csv_file.parent / "visualizations" / f"face_row_{args.row}.png")

    output_file = draw_face(row, output_path)
    print(f"Saved face visualization to: {output_file}")
    print(f"Active labels: {row.get('active_labels', '')}")


if __name__ == "__main__":
    main()
