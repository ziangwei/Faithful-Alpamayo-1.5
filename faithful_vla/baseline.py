"""Local-safe helpers for baseline inference scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL manifest records without importing model or dataset packages."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        _validate_manifest_record(record, path, line_number)
        records.append(record)
    return records


def _validate_manifest_record(record: dict[str, Any], path: Path, line_number: int) -> None:
    for key in ("clip_id", "split", "t0_us"):
        if key not in record:
            raise ValueError(f"{path}:{line_number} missing required field: {key}")
    if not isinstance(record["clip_id"], str) or not record["clip_id"].strip():
        raise ValueError(f"{path}:{line_number} clip_id must be a non-empty string")
    if record["split"] not in {"train", "val", "test"}:
        raise ValueError(f"{path}:{line_number} split must be train, val, or test")
    if not isinstance(record["t0_us"], int):
        raise ValueError(f"{path}:{line_number} t0_us must be an integer")


def select_manifest_records(
    records: list[dict[str, Any]],
    split: str = "val",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Select records by split, preserving manifest order."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    selected = [record for record in records if record["split"] == split]
    if limit is not None:
        selected = selected[:limit]
    return selected


def make_sample_id(record: dict[str, Any], sample_index: int = 0) -> str:
    """Return a stable sample ID for one decision sample."""
    return f"{record['clip_id']}__{record['t0_us']}__{sample_index}"


def make_trajectory_npz_key(
    record: dict[str, Any],
    sample_index: int = 0,
    trajectory_sample_id: int = 0,
) -> str:
    """Return a stable NPZ key prefix for one predicted trajectory sample."""
    return f"{make_sample_id(record, sample_index)}__traj_{trajectory_sample_id}"


def build_dry_run_summary(
    records: list[dict[str, Any]],
    split: str,
    execute: bool,
) -> dict[str, Any]:
    """Build a pre-run summary that is explicit about planned side effects."""
    return {
        "split": split,
        "selected_records": len(records),
        "execute": execute,
        "model_load_planned": execute,
        "dataset_load_planned": execute,
        "download_performed_by_script": False,
        "preview_clip_ids": [record["clip_id"] for record in records[:5]],
    }


def build_prediction_row(
    record: dict[str, Any],
    sample_index: int,
    trajectory_sample_id: int,
    cot: str,
    meta_action: str,
    answer: str,
    runtime_sec: float,
    max_cuda_memory_gb: float | None,
) -> dict[str, Any]:
    """Build one JSONL prediction row for a trajectory sample."""
    sample_id = make_sample_id(record, sample_index)
    trajectory_npz_key = make_trajectory_npz_key(record, sample_index, trajectory_sample_id)
    return {
        "clip_id": record["clip_id"],
        "sample_id": sample_id,
        "trajectory_sample_id": trajectory_sample_id,
        "trajectory_npz_key": trajectory_npz_key,
        "t0_us": record["t0_us"],
        "split": record["split"],
        "nav_text": record.get("nav_text"),
        "required_cameras": list(record.get("required_cameras", [])),
        "cot": cot,
        "meta_action": meta_action,
        "answer": answer,
        "runtime_sec": runtime_sec,
        "max_cuda_memory_gb": max_cuda_memory_gb,
    }


def text_extra_value(extra: dict[str, Any], key: str, set_index: int, traj_index: int) -> str:
    """Extract a text field from the model's nested extra output."""
    value = extra.get(key)
    if value is None:
        return ""
    try:
        return str(value[0][set_index][traj_index])
    except (IndexError, TypeError):
        return ""
