"""Synthetic facial proportion dataset generation."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_ORDER = [
    "middle_third_to_lower",
    "middle_lower_max_to_min",
    "fifth_1_to_2",
    "fifth_2_to_3",
    "fifth_3_to_4",
    "fifth_4_to_5",
    "fifth_1_to_5",
    "fifths_max_to_min",
    "nose_center_offset_ratio",
    "nose_vertical_position_ratio",
    "mouth_center_offset_ratio",
    "mouth_vertical_position_ratio",
    "left_nose_to_left_inner_eye_offset_ratio",
    "right_nose_to_right_inner_eye_offset_ratio",
    "nose_width_to_inner_eye_gap_ratio",
    "nose_width_to_face_width_ratio",
    "mouth_width_to_face_width_ratio",
    "both_eyes_width_to_face_width_ratio",
    "left_eye_width_to_right_eye_width_ratio",
    "left_eye_height_to_right_eye_height_ratio",
    "left_eye_center_y_minus_right_eye_center_y_ratio",
    "left_mouth_corner_y_minus_right_mouth_corner_y_ratio",
]

LABEL_NAMES = [
    "balanced",
    "middle_lower_vertical_imbalance",
    "nose_misaligned",
    "nose_too_high",
    "nose_too_low",
    "mouth_misaligned",
    "mouth_too_high",
    "mouth_too_low",
    "nose_too_narrow_relative_to_inner_eye_gap",
    "nose_too_wide_relative_to_inner_eye_gap",
    "nose_appears_large",
    "mouth_appears_wide",
    "both_eyes_appear_large",
    "both_eyes_appear_small",
    "left_eye_appears_larger",
    "right_eye_appears_larger",
    "left_eye_too_high",
    "right_eye_too_high",
    "left_mouth_corner_too_high",
    "right_mouth_corner_too_high",
]

GENERATED_LABELS = [
    "balanced",
    "middle_lower_vertical_imbalance",
    "nose_misaligned",
    "nose_too_high",
    "nose_too_low",
    "mouth_misaligned",
    "mouth_too_high",
    "mouth_too_low",
    "nose_too_narrow_relative_to_inner_eye_gap",
    "nose_too_wide_relative_to_inner_eye_gap",
    "nose_appears_large",
    "mouth_appears_wide",
    "both_eyes_appear_large",
    "both_eyes_appear_small",
    "left_eye_appears_larger",
    "right_eye_appears_larger",
    "left_eye_too_high",
    "right_eye_too_high",
    "left_mouth_corner_too_high",
    "right_mouth_corner_too_high",
]

RULE_THRESHOLDS = {
    "middle_lower_vertical_imbalance": 1.25,
    "nose_misaligned": 0.04,
    "nose_too_high": 0.43,
    "nose_too_low": 0.57,
    "mouth_misaligned": 0.04,
    "mouth_too_high": 0.68,
    "mouth_too_low": 0.86,
    "nose_too_narrow_relative_to_inner_eye_gap": 0.75,
    "nose_too_wide_relative_to_inner_eye_gap": 1.25,
    "nose_appears_large": 0.24,
    "mouth_appears_wide": 0.42,
    "both_eyes_appear_large": 0.46,
    "both_eyes_appear_small": 0.28,
    "left_eye_large_upper": 1.12,
    "left_eye_large_lower": 0.89,
    "eye_height_offset": 0.02,
    "mouth_corner_offset": 0.025,
}

LABEL_SAMPLE_MULTIPLIERS = {
    "balanced": 1.0,
    "middle_lower_vertical_imbalance": 2.0,
    "nose_misaligned": 2.0,
    "nose_too_high": 2.0,
    "nose_too_low": 2.0,
    "mouth_misaligned": 2.0,
    "mouth_too_high": 2.0,
    "mouth_too_low": 2.0,
    "nose_too_narrow_relative_to_inner_eye_gap": 2.0,
    "nose_too_wide_relative_to_inner_eye_gap": 2.0,
    "nose_appears_large": 2.0,
    "mouth_appears_wide": 2.0,
    "both_eyes_appear_large": 2.0,
    "both_eyes_appear_small": 2.0,
    "left_eye_appears_larger": 2.0,
    "right_eye_appears_larger": 2.0,
    "left_eye_too_high": 2.0,
    "right_eye_too_high": 2.0,
    "left_mouth_corner_too_high": 2.0,
    "right_mouth_corner_too_high": 2.0,
}

FOCUS_LABELS = {
    "nose_too_narrow_relative_to_inner_eye_gap",
    "nose_too_wide_relative_to_inner_eye_gap",
    "nose_appears_large",
    "both_eyes_appear_large",
    "both_eyes_appear_small",
    "nose_too_high",
    "nose_too_low",
    "mouth_too_high",
    "mouth_too_low",
    "left_eye_appears_larger",
    "right_eye_appears_larger",
    "left_eye_too_high",
    "right_eye_too_high",
}

BROAD_IMBALANCE_LABELS = {
    "middle_lower_vertical_imbalance",
}

LABEL_RELEVANT_FEATURES = {
    "middle_lower_vertical_imbalance": [
        "middle_third_to_lower",
        "middle_lower_max_to_min",
    ],
    "nose_misaligned": ["nose_center_offset_ratio"],
    "nose_too_high": ["nose_vertical_position_ratio"],
    "nose_too_low": ["nose_vertical_position_ratio"],
    "mouth_misaligned": ["mouth_center_offset_ratio"],
    "mouth_too_high": ["mouth_vertical_position_ratio"],
    "mouth_too_low": ["mouth_vertical_position_ratio"],
    "nose_too_narrow_relative_to_inner_eye_gap": [
        "nose_width_to_inner_eye_gap_ratio",
        "nose_width_to_face_width_ratio",
    ],
    "nose_too_wide_relative_to_inner_eye_gap": [
        "nose_width_to_inner_eye_gap_ratio",
        "nose_width_to_face_width_ratio",
    ],
    "nose_appears_large": [
        "nose_width_to_face_width_ratio",
        "nose_width_to_inner_eye_gap_ratio",
    ],
    "mouth_appears_wide": ["mouth_width_to_face_width_ratio"],
    "both_eyes_appear_large": [
        "both_eyes_width_to_face_width_ratio",
        "left_eye_width_to_right_eye_width_ratio",
        "left_eye_height_to_right_eye_height_ratio",
    ],
    "both_eyes_appear_small": [
        "both_eyes_width_to_face_width_ratio",
        "left_eye_width_to_right_eye_width_ratio",
        "left_eye_height_to_right_eye_height_ratio",
    ],
    "left_eye_appears_larger": [
        "left_eye_width_to_right_eye_width_ratio",
        "left_eye_height_to_right_eye_height_ratio",
    ],
    "right_eye_appears_larger": [
        "left_eye_width_to_right_eye_width_ratio",
        "left_eye_height_to_right_eye_height_ratio",
    ],
    "left_eye_too_high": ["left_eye_center_y_minus_right_eye_center_y_ratio"],
    "right_eye_too_high": ["left_eye_center_y_minus_right_eye_center_y_ratio"],
    "left_mouth_corner_too_high": ["left_mouth_corner_y_minus_right_mouth_corner_y_ratio"],
    "right_mouth_corner_too_high": ["left_mouth_corner_y_minus_right_mouth_corner_y_ratio"],
}


def validate_label_names() -> None:
    """Validate that the generator labels are fully represented and unique."""
    if len(LABEL_NAMES) != len(set(LABEL_NAMES)):
        raise ValueError("LABEL_NAMES contains duplicate entries.")

    missing_labels = sorted(set(GENERATED_LABELS) - set(LABEL_NAMES))
    if missing_labels:
        raise ValueError(
            "LABEL_NAMES is missing generated labels: "
            + ", ".join(missing_labels)
        )


def canonical_feature_vector() -> dict[str, float]:
    """Return the baseline feature vector for a balanced face."""
    return {
        "middle_third_to_lower": 1.0,
        "middle_lower_max_to_min": 1.0,
        "fifth_1_to_2": 1.0,
        "fifth_2_to_3": 1.0,
        "fifth_3_to_4": 1.0,
        "fifth_4_to_5": 1.0,
        "fifth_1_to_5": 1.0,
        "fifths_max_to_min": 1.0,
        "nose_center_offset_ratio": 0.0,
        "nose_vertical_position_ratio": 0.5,
        "mouth_center_offset_ratio": 0.0,
        "mouth_vertical_position_ratio": 0.77,
        "left_nose_to_left_inner_eye_offset_ratio": 0.035,
        "right_nose_to_right_inner_eye_offset_ratio": 0.035,
        "nose_width_to_inner_eye_gap_ratio": 1.0,
        "nose_width_to_face_width_ratio": 0.2,
        "mouth_width_to_face_width_ratio": 0.36,
        "both_eyes_width_to_face_width_ratio": 0.38,
        "left_eye_width_to_right_eye_width_ratio": 1.0,
        "left_eye_height_to_right_eye_height_ratio": 1.0,
        "left_eye_center_y_minus_right_eye_center_y_ratio": 0.0,
        "left_mouth_corner_y_minus_right_mouth_corner_y_ratio": 0.0,
    }


def _add_small_noise(
    feature_dict: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    """Add light noise so synthetic data is not perfectly uniform."""
    noisy = feature_dict.copy()
    for name in FEATURE_ORDER:
        if "offset_ratio" in name or "minus" in name:
            noisy[name] += float(rng.normal(0.0, 0.005))
        else:
            noisy[name] += float(rng.normal(0.0, 0.015))
    return noisy


def _add_issue_noise(
    feature_dict: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    """Add very small noise for non-balanced samples."""
    noisy = feature_dict.copy()
    for name in FEATURE_ORDER:
        if "offset_ratio" in name or "minus" in name:
            noisy[name] += float(rng.normal(0.0, 0.002))
        else:
            noisy[name] += float(rng.normal(0.0, 0.004))
    return noisy


def _vary_relevant_features(
    feature_dict: dict[str, float], active_labels: list[str], rng: np.random.Generator
) -> dict[str, float]:
    """Add only a little extra variation to relevant active-label features."""
    varied = feature_dict.copy()
    relevant_features: set[str] = set()
    for label in active_labels:
        relevant_features.update(LABEL_RELEVANT_FEATURES.get(label, []))

    for name in relevant_features:
        if "offset_ratio" in name or "minus" in name:
            varied[name] += float(rng.normal(0.0, 0.006))
        else:
            varied[name] += float(rng.normal(0.0, 0.015))

    return varied


def _add_balanced_variation(
    feature_dict: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    """Keep balanced samples safely inside the non-issue region of the rules."""
    varied = feature_dict.copy()

    varied["middle_third_to_lower"] = float(rng.uniform(0.94, 1.16))
    varied["middle_lower_max_to_min"] = max(
        varied["middle_third_to_lower"],
        1.0 / max(varied["middle_third_to_lower"], 1e-6),
    )

    varied["fifth_1_to_2"] = float(rng.uniform(0.9, 1.1))
    varied["fifth_2_to_3"] = float(rng.uniform(0.9, 1.1))
    varied["fifth_3_to_4"] = float(rng.uniform(0.9, 1.1))
    varied["fifth_4_to_5"] = float(rng.uniform(0.9, 1.1))
    varied["fifth_1_to_5"] = float(rng.uniform(0.9, 1.1))
    varied["fifths_max_to_min"] = float(rng.uniform(1.0, 1.25))

    varied["nose_center_offset_ratio"] = float(rng.uniform(0.0, 0.02))
    varied["nose_vertical_position_ratio"] = float(rng.uniform(0.45, 0.55))
    varied["mouth_center_offset_ratio"] = float(rng.uniform(0.0, 0.02))
    varied["mouth_vertical_position_ratio"] = float(rng.triangular(0.68, 0.71, 0.81))
    varied["left_nose_to_left_inner_eye_offset_ratio"] = float(
        rng.uniform(0.0, 0.025)
    )
    varied["right_nose_to_right_inner_eye_offset_ratio"] = float(
        rng.uniform(0.0, 0.025)
    )

    varied["nose_width_to_inner_eye_gap_ratio"] = float(rng.uniform(0.82, 1.15))
    varied["nose_width_to_face_width_ratio"] = float(rng.uniform(0.18, 0.235))
    varied["mouth_width_to_face_width_ratio"] = float(rng.uniform(0.32, 0.40))
    varied["both_eyes_width_to_face_width_ratio"] = float(rng.uniform(0.31, 0.44))
    varied["left_eye_width_to_right_eye_width_ratio"] = float(rng.uniform(0.93, 1.07))
    varied["left_eye_height_to_right_eye_height_ratio"] = float(
        rng.uniform(0.93, 1.07)
    )
    varied["left_eye_center_y_minus_right_eye_center_y_ratio"] = float(
        rng.uniform(-0.018, 0.018)
    )
    varied["left_mouth_corner_y_minus_right_mouth_corner_y_ratio"] = float(
        rng.uniform(-0.022, 0.022)
    )

    return varied


def _clip_feature_values(feature_dict: dict[str, float]) -> dict[str, float]:
    """Keep ratios positive and offsets within a reasonable normalized range."""
    clipped = feature_dict.copy()
    for name in FEATURE_ORDER:
        if "minus" in name:
            clipped[name] = float(np.clip(clipped[name], -0.25, 0.25))
        elif "offset_ratio" in name:
            clipped[name] = float(np.clip(clipped[name], 0.0, 0.5))
        else:
            clipped[name] = float(max(clipped[name], 0.05))
    return clipped


def _sample_above_threshold(
    rng: np.random.Generator, threshold: float, margin_low: float, margin_high: float
) -> float:
    """Sample a value clearly above a threshold."""
    return float(threshold + rng.uniform(margin_low, margin_high))


def _sample_below_threshold(
    rng: np.random.Generator, threshold: float, margin_low: float, margin_high: float
) -> float:
    """Sample a value clearly below a threshold."""
    return float(max(threshold - rng.uniform(margin_low, margin_high), 0.05))


def _sample_soft_above_threshold(
    rng: np.random.Generator,
    threshold: float,
    below_margin: float,
    above_margin: float,
    below_probability: float = 0.0,
) -> float:
    """Usually sample above threshold, but sometimes just below it."""
    if float(rng.random()) < below_probability:
        return _sample_below_threshold(rng, threshold, below_margin * 0.3, below_margin)
    return _sample_above_threshold(rng, threshold, above_margin * 0.3, above_margin)


def _sample_soft_below_threshold(
    rng: np.random.Generator,
    threshold: float,
    below_margin: float,
    above_margin: float,
    above_probability: float = 0.0,
) -> float:
    """Usually sample below threshold, but sometimes just above it."""
    if float(rng.random()) < above_probability:
        return _sample_above_threshold(rng, threshold, above_margin * 0.3, above_margin)
    return _sample_below_threshold(rng, threshold, below_margin * 0.3, below_margin)


def _apply_label_effect(
    feature_dict: dict[str, float], label: str, rng: np.random.Generator
) -> None:
    """Apply one label-specific perturbation to the feature dictionary."""
    if label == "balanced":
        return
    if label == "middle_lower_vertical_imbalance":
        target = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["middle_lower_vertical_imbalance"], 0.06, 0.28
        )
        feature_dict["middle_third_to_lower"] = target
        feature_dict["middle_lower_max_to_min"] = target
        return
    if label == "nose_misaligned":
        feature_dict["nose_center_offset_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["nose_misaligned"], 0.03, 0.14
        )
        return
    if label == "nose_too_high":
        feature_dict["nose_vertical_position_ratio"] = _sample_soft_below_threshold(
            rng, RULE_THRESHOLDS["nose_too_high"], 0.03, 0.12
        )
        return
    if label == "nose_too_low":
        feature_dict["nose_vertical_position_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["nose_too_low"], 0.05, 0.16
        )
        return
    if label == "mouth_misaligned":
        feature_dict["mouth_center_offset_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["mouth_misaligned"], 0.03, 0.15
        )
        return
    if label == "mouth_too_high":
        feature_dict["mouth_vertical_position_ratio"] = _sample_soft_below_threshold(
            rng, RULE_THRESHOLDS["mouth_too_high"], 0.08, 0.16
        )
        return
    if label == "mouth_too_low":
        feature_dict["mouth_vertical_position_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["mouth_too_low"], 0.04, 0.14
        )
        return
    if label == "nose_too_narrow_relative_to_inner_eye_gap":
        feature_dict["nose_width_to_inner_eye_gap_ratio"] = _sample_soft_below_threshold(
            rng, RULE_THRESHOLDS["nose_too_narrow_relative_to_inner_eye_gap"], 0.14, 0.32
        )
        feature_dict["nose_width_to_face_width_ratio"] = float(rng.uniform(0.08, 0.145))
        return
    if label == "nose_too_wide_relative_to_inner_eye_gap":
        feature_dict["nose_width_to_inner_eye_gap_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["nose_too_wide_relative_to_inner_eye_gap"], 0.1, 0.36
        )
        feature_dict["nose_width_to_face_width_ratio"] = float(rng.uniform(0.25, 0.33))
        return
    if label == "nose_appears_large":
        feature_dict["nose_width_to_face_width_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["nose_appears_large"], 0.05, 0.12
        )
        feature_dict["nose_width_to_inner_eye_gap_ratio"] = float(rng.uniform(1.12, 1.32))
        return
    if label == "mouth_appears_wide":
        feature_dict["mouth_width_to_face_width_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["mouth_appears_wide"], 0.04, 0.14
        )
        return
    if label == "both_eyes_appear_large":
        feature_dict["both_eyes_width_to_face_width_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["both_eyes_appear_large"], 0.02, 0.1
        )
        feature_dict["left_eye_width_to_right_eye_width_ratio"] = float(
            rng.uniform(0.985, 1.015)
        )
        feature_dict["left_eye_height_to_right_eye_height_ratio"] = float(
            rng.uniform(0.985, 1.015)
        )
        return
    if label == "both_eyes_appear_small":
        feature_dict["both_eyes_width_to_face_width_ratio"] = _sample_soft_below_threshold(
            rng, RULE_THRESHOLDS["both_eyes_appear_small"], 0.02, 0.1
        )
        feature_dict["left_eye_width_to_right_eye_width_ratio"] = float(
            rng.uniform(0.985, 1.015)
        )
        feature_dict["left_eye_height_to_right_eye_height_ratio"] = float(
            rng.uniform(0.985, 1.015)
        )
        return
    if label == "left_eye_appears_larger":
        feature_dict["left_eye_width_to_right_eye_width_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["left_eye_large_upper"], 0.05, 0.24
        )
        feature_dict["left_eye_height_to_right_eye_height_ratio"] = _sample_soft_above_threshold(
            rng, RULE_THRESHOLDS["left_eye_large_upper"], 0.05, 0.24
        )
        return
    if label == "right_eye_appears_larger":
        feature_dict["left_eye_width_to_right_eye_width_ratio"] = _sample_soft_below_threshold(
            rng, RULE_THRESHOLDS["left_eye_large_lower"], 0.04, 0.22
        )
        feature_dict["left_eye_height_to_right_eye_height_ratio"] = _sample_soft_below_threshold(
            rng, RULE_THRESHOLDS["left_eye_large_lower"], 0.04, 0.22
        )
        return
    if label == "left_eye_too_high":
        feature_dict["left_eye_center_y_minus_right_eye_center_y_ratio"] = -(
            RULE_THRESHOLDS["eye_height_offset"] + float(rng.uniform(0.04, 0.12))
        )
        return
    if label == "right_eye_too_high":
        feature_dict["left_eye_center_y_minus_right_eye_center_y_ratio"] = (
            RULE_THRESHOLDS["eye_height_offset"] + float(rng.uniform(0.04, 0.12))
        )
        return
    if label == "left_mouth_corner_too_high":
        feature_dict["left_mouth_corner_y_minus_right_mouth_corner_y_ratio"] = -(
            RULE_THRESHOLDS["mouth_corner_offset"] + float(rng.uniform(0.02, 0.1))
        )
        return
    if label == "right_mouth_corner_too_high":
        feature_dict["left_mouth_corner_y_minus_right_mouth_corner_y_ratio"] = (
            RULE_THRESHOLDS["mouth_corner_offset"] + float(rng.uniform(0.02, 0.1))
        )
        return

    raise ValueError(f"Unsupported synthetic label: {label}")


def _sample_count_for_label(label: str, samples_per_class: int) -> int:
    """Return a label-specific sample count for better class emphasis."""
    multiplier = LABEL_SAMPLE_MULTIPLIERS.get(label, 1.0)
    return max(1, int(round(samples_per_class * multiplier)))


def _select_multilabel_combos(
    non_balanced_labels: list[str],
    samples_per_multilabel_combo: int,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """Prefer multi-label combos that teach more specific local issues."""
    all_combos = [
        combo
        for combo_size in (2, 3)
        for combo in combinations(non_balanced_labels, combo_size)
    ]
    rng.shuffle(all_combos)

    def combo_priority(combo: tuple[str, ...]) -> tuple[int, int, int]:
        focus_count = sum(label in FOCUS_LABELS for label in combo)
        broad_count = sum(label in BROAD_IMBALANCE_LABELS for label in combo)
        eye_specific_count = sum(
            label
            in {
                "both_eyes_appear_large",
                "both_eyes_appear_small",
                "left_eye_appears_larger",
                "right_eye_appears_larger",
                "left_eye_too_high",
                "right_eye_too_high",
            }
            for label in combo
        )
        return (focus_count, eye_specific_count, -broad_count)

    sorted_combos = sorted(all_combos, key=combo_priority, reverse=True)
    combo_count = min(samples_per_multilabel_combo, len(sorted_combos))
    return sorted_combos[:combo_count]


def _label_columns(active_labels: list[str]) -> dict[str, int]:
    """Create one binary label column per label name."""
    active = set(active_labels)
    return {
        f"label_{label}": int(label in active)
        for label in LABEL_NAMES
    }


def generate_sample(
    active_labels: list[str], rng: np.random.Generator
) -> dict[str, float | int | str]:
    """Generate one synthetic sample with one or more active labels."""
    validate_label_names()

    if not active_labels:
        raise ValueError("active_labels must contain at least one label.")

    if len(set(active_labels)) != len(active_labels):
        raise ValueError("active_labels contains duplicate labels.")

    invalid_labels = sorted(set(active_labels) - set(LABEL_NAMES))
    if invalid_labels:
        raise ValueError(
            "Active labels are not present in LABEL_NAMES: "
            + ", ".join(invalid_labels)
        )

    if "balanced" in active_labels and len(active_labels) > 1:
        raise ValueError("'balanced' cannot be combined with other active labels.")

    if active_labels == ["balanced"]:
        features = canonical_feature_vector()
        features = _add_balanced_variation(features, rng)
        features = _clip_feature_values(features)
    else:
        features = _add_issue_noise(canonical_feature_vector(), rng)
        for label in active_labels:
            _apply_label_effect(features, label, rng)
        features = _vary_relevant_features(features, active_labels, rng)
        features = _clip_feature_values(features)

    row: dict[str, float | int | str] = {}
    for name in FEATURE_ORDER:
        row[name] = float(features[name])

    label_columns = _label_columns(active_labels)
    if any(label != "balanced" and value == 1 for label, value in zip(LABEL_NAMES, label_columns.values(), strict=False)):
        label_columns["label_balanced"] = 0
    elif active_labels == ["balanced"]:
        label_columns["label_balanced"] = 1

    row.update(label_columns)
    row["active_labels"] = "|".join(active_labels)
    return row


def generate_dataset(
    samples_per_class: int,
    samples_per_multilabel_combo: int,
    random_state: int = 42,
) -> dict[str, Any]:
    """Generate single-label, balanced, and random multi-label synthetic data."""
    validate_label_names()

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int | str]] = []
    non_balanced_labels = [label for label in LABEL_NAMES if label != "balanced"]

    balanced_count = _sample_count_for_label("balanced", samples_per_class)
    for _ in range(balanced_count):
        rows.append(generate_sample(["balanced"], rng))

    single_label_issue_rows = 0
    for label in non_balanced_labels:
        label_count = _sample_count_for_label(label, samples_per_class)
        single_label_issue_rows += label_count
        for _ in range(label_count):
            rows.append(generate_sample([label], rng))

    selected_combos = _select_multilabel_combos(
        non_balanced_labels, samples_per_multilabel_combo, rng
    )
    target_combo_rows = max(1, int(round(single_label_issue_rows * 0.25)))
    combo_rows_per_combo = max(1, target_combo_rows // max(len(selected_combos), 1))
    for combo in selected_combos:
        for _ in range(combo_rows_per_combo):
            rows.append(generate_sample(list(combo), rng))

    feature_names = FEATURE_ORDER.copy()
    label_columns = [f"label_{label}" for label in LABEL_NAMES]

    return {
        "rows": rows,
        "feature_names": feature_names,
        "label_names": LABEL_NAMES.copy(),
        "label_columns": label_columns,
        "columns": feature_names + label_columns + ["active_labels"],
    }


def save_dataset_csv(
    output_path: str,
    samples_per_class: int = 200,
    samples_per_multilabel_combo: int = 50,
    random_state: int = 42,
) -> Path:
    """Generate the dataset and save it as a CSV file."""
    dataset = generate_dataset(
        samples_per_class=samples_per_class,
        samples_per_multilabel_combo=samples_per_multilabel_combo,
        random_state=random_state,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset["columns"])
        writer.writeheader()
        writer.writerows(dataset["rows"])

    return output_file


def main() -> None:
    """Generate and save a synthetic SketchCritic CSV dataset."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SketchCritic CSV dataset."
    )
    parser.add_argument("output_path", help="Path to the CSV file to write.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=200,
        help="Number of balanced and single-label samples per label.",
    )
    parser.add_argument(
        "--samples-per-multilabel-combo",
        type=int,
        default=50,
        help="Number of samples to generate for each selected multi-label combo.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    args = parser.parse_args()

    output_file = save_dataset_csv(
        output_path=args.output_path,
        samples_per_class=args.samples_per_class,
        samples_per_multilabel_combo=args.samples_per_multilabel_combo,
        random_state=args.random_state,
    )
    print(f"Saved synthetic dataset to: {output_file}")


if __name__ == "__main__":
    main()
