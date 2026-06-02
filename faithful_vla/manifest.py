"""Manifest helpers that do not import dataset or model packages."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from faithful_vla.constants import (
    DEFAULT_CAMERA_FEATURES,
    DEFAULT_NUM_DECISION_SAMPLES,
    DEFAULT_SPLIT_COUNTS,
    DEFAULT_T0_US,
)


@dataclass(frozen=True)
class ManifestRecord:
    """One clip-level manifest record."""

    clip_id: str
    split: str
    t0_us: int
    num_decision_samples: int
    required_cameras: list[str]
    required_labels: list[str]
    scene_tags: list[str]
    local_paths: dict[str, str]


def read_clip_ids(path: Path) -> list[str]:
    """Read clip IDs from txt, csv, or jsonl without requiring pandas."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl_clip_ids(path)
    if suffix == ".csv":
        return _read_csv_clip_ids(path)
    return _read_text_clip_ids(path)


def _read_text_clip_ids(path: Path) -> list[str]:
    return _dedupe_preserve_order(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
    )


def _read_jsonl_clip_ids(path: Path) -> list[str]:
    values: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        clip_id = payload.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise ValueError(f"{path}:{line_number} missing non-empty clip_id")
        values.append(clip_id.strip())
    return _dedupe_preserve_order(values)


def _read_csv_clip_ids(path: Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames and "clip_id" in reader.fieldnames:
            for row in reader:
                values.append(row["clip_id"].strip())
        else:
            stream.seek(0)
            plain_reader = csv.reader(stream)
            for row in plain_reader:
                if row:
                    values.append(row[0].strip())
    return _dedupe_preserve_order(values)


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def assign_clip_splits(
    clip_ids: list[str],
    split_counts: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, str]:
    """Assign clip-level splits deterministically."""
    split_counts = split_counts or DEFAULT_SPLIT_COUNTS
    required = sum(split_counts.values())
    if len(clip_ids) < required:
        raise ValueError(f"Need at least {required} clip IDs, got {len(clip_ids)}")

    shuffled = list(clip_ids)
    random.Random(seed).shuffle(shuffled)

    assignments: dict[str, str] = {}
    cursor = 0
    for split in ("train", "val", "test"):
        count = split_counts[split]
        for clip_id in shuffled[cursor : cursor + count]:
            assignments[clip_id] = split
        cursor += count
    return assignments


def build_manifest_records(
    clip_ids: list[str],
    assignments: dict[str, str],
    t0_us: int = DEFAULT_T0_US,
    num_decision_samples: int = DEFAULT_NUM_DECISION_SAMPLES,
    required_cameras: list[str] | None = None,
) -> list[ManifestRecord]:
    """Build manifest records from split assignments."""
    required_cameras = required_cameras or DEFAULT_CAMERA_FEATURES
    records: list[ManifestRecord] = []
    for clip_id in clip_ids:
        split = assignments.get(clip_id)
        if split is None:
            continue
        records.append(
            ManifestRecord(
                clip_id=clip_id,
                split=split,
                t0_us=t0_us,
                num_decision_samples=num_decision_samples,
                required_cameras=list(required_cameras),
                required_labels=["EGOMOTION"],
                scene_tags=[],
                local_paths={},
            )
        )
    return records


def summarize_splits(records: list[ManifestRecord]) -> dict[str, object]:
    """Summarize split counts and leakage checks."""
    counts = {"train": 0, "val": 0, "test": 0}
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    leakage: list[dict[str, str]] = []
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
        previous = seen.get(record.clip_id)
        if previous is None:
            seen[record.clip_id] = record.split
        elif previous == record.split:
            duplicates.append(record.clip_id)
        else:
            leakage.append({"clip_id": record.clip_id, "first_split": previous, "split": record.split})

    return {
        "counts": counts,
        "num_records": len(records),
        "num_unique_clips": len(seen),
        "duplicate_clip_ids": sorted(set(duplicates)),
        "split_leakage": leakage,
        "split_unit": "clip",
    }


def write_manifest(records: list[ManifestRecord], output_path: Path) -> None:
    """Write JSONL manifest records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def write_json(payload: dict[str, object], output_path: Path) -> None:
    """Write stable, human-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

