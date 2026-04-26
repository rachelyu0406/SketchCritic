"""Feature computation from MediaPipe face landmarks."""

from __future__ import annotations

from typing import Any


LEFT_BROW_INDEX = 105
RIGHT_BROW_INDEX = 334
NOSE_BASE_INDEX = 2
LEFT_NOSE_SIDE_INDEX = 98
RIGHT_NOSE_SIDE_INDEX = 327
CHIN_INDEX = 152

LEFT_FACE_EDGE_INDEX = 234
RIGHT_FACE_EDGE_INDEX = 454

LEFT_EYE_OUTER_INDEX = 33
LEFT_EYE_INNER_INDEX = 133
LEFT_EYE_UPPER_INDEX = 159
LEFT_EYE_LOWER_INDEX = 145
RIGHT_EYE_INNER_INDEX = 362
RIGHT_EYE_OUTER_INDEX = 263
RIGHT_EYE_UPPER_INDEX = 386
RIGHT_EYE_LOWER_INDEX = 374

LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

MIN_SECTION_WIDTH = 1e-6

# Convert the landmark list into a dictionary keyed by landmark index.
def _point_map(landmark_result: dict[str, Any]) -> dict[int, dict[str, float]]:
    """Convert the structured landmark list into an index lookup map."""
    points = landmark_result["landmarks"]
    return {
        int(point["index"]): {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "z": float(point["z"]),
        }
        for point in points
    }

# Compute the mean for small groups of related coordinates.
def _average(values: list[float]) -> float:
    """Return the arithmetic mean of a list of numbers."""
    return sum(values) / len(values)

# Avoid division-by-zero when turning raw distances into ratios.
def _safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a ratio while avoiding division by zero."""
    epsilon = 1e-6
    return numerator / max(abs(denominator), epsilon)

# Fetch a required landmark point and fail with a clear message if it is missing.
def require_point(
    points: dict[int, dict[str, float]], index: int, name: str
) -> dict[str, float]:
    """Return a landmark point or raise a clear error if it is missing."""
    point = points.get(index)
    if point is None:
        raise ValueError(f"Missing required landmark '{name}' at index {index}.")
    return point

# Derive the normalized facial proportion features used by the classifiers.
def compute_features(landmark_result: dict[str, Any]) -> dict[str, float]:
    """Compute facial proportion features from one-face landmark output."""
    points = _point_map(landmark_result)

    left_brow = require_point(points, LEFT_BROW_INDEX, "left_brow")
    right_brow = require_point(points, RIGHT_BROW_INDEX, "right_brow")
    nose_base = require_point(points, NOSE_BASE_INDEX, "nose_base")
    left_nose_side = require_point(points, LEFT_NOSE_SIDE_INDEX, "left_nose_side")
    right_nose_side = require_point(points, RIGHT_NOSE_SIDE_INDEX, "right_nose_side")
    chin = require_point(points, CHIN_INDEX, "chin")
    left_face_edge = require_point(points, LEFT_FACE_EDGE_INDEX, "left_face_edge")
    right_face_edge = require_point(points, RIGHT_FACE_EDGE_INDEX, "right_face_edge")
    left_eye_outer = require_point(points, LEFT_EYE_OUTER_INDEX, "left_eye_outer")
    left_eye_inner = require_point(points, LEFT_EYE_INNER_INDEX, "left_eye_inner")
    left_eye_upper = require_point(points, LEFT_EYE_UPPER_INDEX, "left_eye_upper")
    left_eye_lower = require_point(points, LEFT_EYE_LOWER_INDEX, "left_eye_lower")
    right_eye_inner = require_point(points, RIGHT_EYE_INNER_INDEX, "right_eye_inner")
    right_eye_outer = require_point(points, RIGHT_EYE_OUTER_INDEX, "right_eye_outer")
    right_eye_upper = require_point(points, RIGHT_EYE_UPPER_INDEX, "right_eye_upper")
    right_eye_lower = require_point(points, RIGHT_EYE_LOWER_INDEX, "right_eye_lower")
    left_mouth_corner = require_point(
        points, LEFT_MOUTH_CORNER_INDEX, "left_mouth_corner"
    )
    right_mouth_corner = require_point(
        points, RIGHT_MOUTH_CORNER_INDEX, "right_mouth_corner"
    )

    brow_y = _average([left_brow["y"], right_brow["y"]])
    nose_base_y = nose_base["y"]
    chin_y = chin["y"]

    middle_third = abs(nose_base_y - brow_y)
    lower_third = abs(chin_y - nose_base_y)

    boundary_xs = sorted(
        [
            left_face_edge["x"],
            left_eye_outer["x"],
            left_eye_inner["x"],
            right_eye_inner["x"],
            right_eye_outer["x"],
            right_face_edge["x"],
        ]
    )
    f1 = boundary_xs[1] - boundary_xs[0]
    f2 = boundary_xs[2] - boundary_xs[1]
    f3 = boundary_xs[3] - boundary_xs[2]
    f4 = boundary_xs[4] - boundary_xs[3]
    f5 = boundary_xs[5] - boundary_xs[4]

    fifths = [f1, f2, f3, f4, f5]
    if any(width <= MIN_SECTION_WIDTH for width in fifths):
        raise ValueError("One or more fifth sections are too small to measure.")

    face_left_x = left_face_edge["x"]
    face_right_x = right_face_edge["x"]
    face_width = face_right_x - face_left_x
    if face_width <= MIN_SECTION_WIDTH:
        raise ValueError("Face width is too small to compute normalized features.")

    face_center_x = (face_left_x + face_right_x) / 2.0
    nose_center_x = _average([left_nose_side["x"], right_nose_side["x"]])
    mouth_center_x = _average([left_mouth_corner["x"], right_mouth_corner["x"]])
    mouth_center_y = _average([left_mouth_corner["y"], right_mouth_corner["y"]])
    nose_width = abs(right_nose_side["x"] - left_nose_side["x"])
    inner_eye_gap = abs(right_eye_inner["x"] - left_eye_inner["x"])
    if inner_eye_gap <= MIN_SECTION_WIDTH:
        raise ValueError("Inner eye gap is too small to compute nose-width ratio.")

    lower_face_height = chin_y - brow_y
    if lower_face_height <= MIN_SECTION_WIDTH:
        raise ValueError("Lower face height is too small to compute vertical features.")

    left_eye_width = abs(left_eye_inner["x"] - left_eye_outer["x"])
    right_eye_width = abs(right_eye_outer["x"] - right_eye_inner["x"])
    left_eye_height = abs(left_eye_lower["y"] - left_eye_upper["y"])
    right_eye_height = abs(right_eye_lower["y"] - right_eye_upper["y"])
    if left_eye_width <= MIN_SECTION_WIDTH or right_eye_width <= MIN_SECTION_WIDTH:
        raise ValueError("Eye width is too small to compute eye-size ratios.")
    if left_eye_height <= MIN_SECTION_WIDTH or right_eye_height <= MIN_SECTION_WIDTH:
        raise ValueError("Eye height is too small to compute eye-size ratios.")

    left_eye_center_y = _average(
        [
            left_eye_inner["y"],
            left_eye_outer["y"],
            left_eye_upper["y"],
            left_eye_lower["y"],
        ]
    )
    right_eye_center_y = _average(
        [
            right_eye_inner["y"],
            right_eye_outer["y"],
            right_eye_upper["y"],
            right_eye_lower["y"],
        ]
    )
    mouth_width = abs(right_mouth_corner["x"] - left_mouth_corner["x"])

    return {
        "middle_third_to_lower": _safe_ratio(middle_third, lower_third),
        "middle_lower_max_to_min": _safe_ratio(
            max(middle_third, lower_third), min(middle_third, lower_third)
        ),
        "fifth_1_to_2": _safe_ratio(f1, f2),
        "fifth_2_to_3": _safe_ratio(f2, f3),
        "fifth_3_to_4": _safe_ratio(f3, f4),
        "fifth_4_to_5": _safe_ratio(f4, f5),
        "fifth_1_to_5": _safe_ratio(f1, f5),
        "fifths_max_to_min": _safe_ratio(max(fifths), min(fifths)),
        "nose_center_offset_ratio": abs(nose_center_x - face_center_x) / face_width,
        "nose_vertical_position_ratio": (nose_base_y - brow_y) / lower_face_height,
        "mouth_center_offset_ratio": abs(mouth_center_x - face_center_x)
        / face_width,
        "mouth_vertical_position_ratio": (mouth_center_y - brow_y) / lower_face_height,
        "left_nose_to_left_inner_eye_offset_ratio": abs(
            left_nose_side["x"] - left_eye_inner["x"]
        )
        / face_width,
        "right_nose_to_right_inner_eye_offset_ratio": abs(
            right_nose_side["x"] - right_eye_inner["x"]
        )
        / face_width,
        "nose_width_to_inner_eye_gap_ratio": nose_width / inner_eye_gap,
        "nose_width_to_face_width_ratio": nose_width / face_width,
        "mouth_width_to_face_width_ratio": mouth_width / face_width,
        "both_eyes_width_to_face_width_ratio": (
            left_eye_width + right_eye_width
        )
        / face_width,
        "left_eye_width_to_right_eye_width_ratio": _safe_ratio(
            left_eye_width, right_eye_width
        ),
        "left_eye_height_to_right_eye_height_ratio": _safe_ratio(
            left_eye_height, right_eye_height
        ),
        "left_eye_center_y_minus_right_eye_center_y_ratio": (
            left_eye_center_y - right_eye_center_y
        )
        / lower_face_height,
        "left_mouth_corner_y_minus_right_mouth_corner_y_ratio": (
            left_mouth_corner["y"] - right_mouth_corner["y"]
        )
        / lower_face_height,
    }
