"""Shared constants for local-safe project scripts."""

DEFAULT_CAMERA_FEATURES = [
    "CAMERA_CROSS_LEFT_120FOV",
    "CAMERA_FRONT_WIDE_120FOV",
    "CAMERA_CROSS_RIGHT_120FOV",
    "CAMERA_FRONT_TELE_30FOV",
]

DEFAULT_SPLIT_COUNTS = {
    "train": 180,
    "val": 60,
    "test": 60,
}

DEFAULT_T0_US = 5_100_000
DEFAULT_NUM_DECISION_SAMPLES = 1

