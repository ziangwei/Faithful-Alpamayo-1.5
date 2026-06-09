"""Metrics for Alpamayo baseline outputs.

The functions in this module intentionally use only the Python standard
library so they can be tested locally without model, dataset, or numpy
dependencies.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence


DEFAULT_INTENT_KEYWORDS: dict[str, list[str]] = {
    "stop": ["stop", "stopped", "full stop"],
    "slow_down": ["slow down", "slowing", "brake", "decelerate"],
    "yield": ["yield", "give way"],
    "avoid": ["avoid", "steer around", "obstacle"],
    "lane_change_left": ["change lane left", "lane change left", "move left"],
    "lane_change_right": ["change lane right", "lane change right", "move right"],
    "turn_left": ["turn left", "left turn"],
    "turn_right": ["turn right", "right turn"],
    "go_straight": ["go straight", "proceed", "continue straight"],
}

DEFAULT_CONSISTENCY_THRESHOLDS: dict[str, float] = {
    "stop_final_speed_mps_max": 0.5,
    "slow_down_speed_delta_mps_min": 1.0,
    "yield_speed_delta_mps_min": 0.5,
    "lane_change_lateral_m_min": 1.0,
    "turn_heading_change_rad_min": 0.35,
    "avoid_adjustment_lateral_m_min": 0.6,
    "avoid_adjustment_speed_delta_mps_min": 0.5,
}


def compute_trajectory_metrics(
    pred_xyz: Sequence[Sequence[float]],
    gt_xyz: Sequence[Sequence[float]],
    time_step: float,
) -> dict[str, float]:
    """Compute trajectory and kinematic metrics from predicted and GT xyz points."""
    pred_xy = _xy_points(pred_xyz)
    gt_xy = _xy_points(gt_xyz)
    if len(pred_xy) != len(gt_xy):
        raise ValueError(f"pred and gt must have the same length, got {len(pred_xy)} and {len(gt_xy)}")
    if len(pred_xy) < 2:
        raise ValueError("trajectory must contain at least two points")
    if time_step <= 0:
        raise ValueError("time_step must be positive")

    distances = [_distance(pred, gt) for pred, gt in zip(pred_xy, gt_xy)]
    speeds = [
        _distance(pred_xy[index], pred_xy[index - 1]) / time_step
        for index in range(1, len(pred_xy))
    ]
    accelerations = [
        (speeds[index] - speeds[index - 1]) / time_step for index in range(1, len(speeds))
    ]
    jerks = [
        (accelerations[index] - accelerations[index - 1]) / time_step
        for index in range(1, len(accelerations))
    ]
    heading_change = _heading_change(pred_xy)

    return {
        "ade_m": _mean(distances),
        "fde_m": distances[-1],
        "average_speed_mps": _mean(speeds),
        "initial_speed_mps": speeds[0],
        "final_speed_mps": speeds[-1],
        "speed_delta_mps": speeds[0] - speeds[-1],
        "max_acceleration_mps2": max(accelerations) if accelerations else 0.0,
        "max_deceleration_mps2": min(accelerations) if accelerations else 0.0,
        "mean_abs_jerk_mps3": _mean([abs(value) for value in jerks]) if jerks else 0.0,
        "lateral_displacement_m": pred_xy[-1][1] - pred_xy[0][1],
        "abs_lateral_displacement_m": abs(pred_xy[-1][1] - pred_xy[0][1]),
        "heading_change_rad": heading_change,
        "abs_heading_change_rad": abs(heading_change),
    }


def parse_intents(
    fields: dict[str, str | None],
    intent_keywords: dict[str, list[str]] | None = None,
) -> list[str]:
    """Parse coarse driving intents from text fields in stable priority order."""
    intent_keywords = intent_keywords or DEFAULT_INTENT_KEYWORDS
    text = " ".join(value or "" for value in fields.values()).lower()
    intents: list[str] = []
    for intent, keywords in intent_keywords.items():
        if any(keyword in text for keyword in keywords):
            intents.append(intent)
    return intents


def check_reasoning_action_consistency(
    intents: Iterable[str],
    behavior: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Check whether parsed reasoning intents match trajectory behavior."""
    thresholds = thresholds or DEFAULT_CONSISTENCY_THRESHOLDS
    failed_checks: list[str] = []
    intents = list(intents)

    if "stop" in intents and behavior["final_speed_mps"] > thresholds["stop_final_speed_mps_max"]:
        failed_checks.append("stop")
    if (
        "slow_down" in intents
        and behavior["speed_delta_mps"] < thresholds["slow_down_speed_delta_mps_min"]
    ):
        failed_checks.append("slow_down")
    if "yield" in intents and behavior["speed_delta_mps"] < thresholds["yield_speed_delta_mps_min"]:
        failed_checks.append("yield")
    if (
        "lane_change_left" in intents
        and behavior["lateral_displacement_m"] < thresholds["lane_change_lateral_m_min"]
    ):
        failed_checks.append("lane_change_left")
    if (
        "lane_change_right" in intents
        and -behavior["lateral_displacement_m"] < thresholds["lane_change_lateral_m_min"]
    ):
        failed_checks.append("lane_change_right")
    if (
        "turn_left" in intents
        and behavior["heading_change_rad"] < thresholds["turn_heading_change_rad_min"]
    ):
        failed_checks.append("turn_left")
    if (
        "turn_right" in intents
        and -behavior["heading_change_rad"] < thresholds["turn_heading_change_rad_min"]
    ):
        failed_checks.append("turn_right")
    if "avoid" in intents:
        lateral_ok = (
            behavior["abs_lateral_displacement_m"]
            >= thresholds["avoid_adjustment_lateral_m_min"]
        )
        speed_ok = behavior["speed_delta_mps"] >= thresholds["avoid_adjustment_speed_delta_mps_min"]
        if not lateral_ok and not speed_ok:
            failed_checks.append("avoid")

    return {
        "is_consistent": not failed_checks,
        "failed_checks": failed_checks,
        "num_intents": len(intents),
    }


def summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample metric rows into a compact summary."""
    if not rows:
        return {
            "num_samples": 0,
            "mean_ade_m": None,
            "mean_fde_m": None,
            "reasoning_action_consistency_rate": None,
            "inconsistency_count_by_type": {},
        }

    failed_counter: Counter[str] = Counter()
    for row in rows:
        failed_counter.update(row.get("failed_checks", []))

    return {
        "num_samples": len(rows),
        "mean_ade_m": _mean_key(rows, "ade_m"),
        "mean_fde_m": _mean_key(rows, "fde_m"),
        "mean_average_speed_mps": _mean_key(rows, "average_speed_mps"),
        "mean_final_speed_mps": _mean_key(rows, "final_speed_mps"),
        "reasoning_action_consistency_rate": _mean(
            1.0 if row["is_consistent"] else 0.0 for row in rows
        ),
        "num_inconsistent": sum(1 for row in rows if not row["is_consistent"]),
        "inconsistency_count_by_type": dict(sorted(failed_counter.items())),
    }


def _xy_points(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    return [(float(point[0]), float(point[1])) for point in points]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_key(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    if not values:
        return None
    return _mean(values)


def _heading_change(points: Sequence[tuple[float, float]]) -> float:
    start_heading = _segment_heading(points[0], points[1])
    end_heading = _segment_heading(points[-2], points[-1])
    return _normalize_angle(end_heading - start_heading)


def _segment_heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
