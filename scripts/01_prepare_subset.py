#!/usr/bin/env python
"""Create a dry-run 300-clip manifest without downloading dataset content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.constants import DEFAULT_SPLIT_COUNTS, DEFAULT_T0_US
from faithful_vla.manifest import (
    assign_clip_splits,
    build_manifest_records,
    read_clip_ids,
    summarize_splits,
    write_json,
    write_manifest,
)


def build_storage_report(num_records: int) -> dict[str, object]:
    """Return a storage report for manifest-only preparation."""
    return {
        "download_performed": False,
        "model_download_performed": False,
        "estimated_additional_dataset_gb": 0.0,
        "estimated_additional_model_gb": 0.0,
        "max_additional_gb_before_stop": 50,
        "records": num_records,
        "note": "Manifest-only dry run. No dataset files, media, or model weights copied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clip-id-file",
        required=True,
        type=Path,
        help="Text, CSV, or JSONL file containing clip IDs. CSV/JSONL may use clip_id field.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=DEFAULT_SPLIT_COUNTS["train"])
    parser.add_argument("--val-count", type=int, default=DEFAULT_SPLIT_COUNTS["val"])
    parser.add_argument("--test-count", type=int, default=DEFAULT_SPLIT_COUNTS["test"])
    parser.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    parser.add_argument("--write", action="store_true", help="Write manifest JSONL and summaries.")
    args = parser.parse_args()

    clip_ids = read_clip_ids(args.clip_id_file)
    split_counts = {
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }
    assignments = assign_clip_splits(clip_ids, split_counts=split_counts, seed=args.seed)
    selected_clip_ids = [clip_id for clip_id in clip_ids if clip_id in assignments]
    records = build_manifest_records(selected_clip_ids, assignments, t0_us=args.t0_us)
    split_summary = summarize_splits(records)
    storage_report = build_storage_report(len(records))

    preview = {
        "input_clip_ids": len(clip_ids),
        "selected_clip_ids": len(records),
        "split_summary": split_summary,
        "storage_report": storage_report,
        "write_enabled": args.write,
    }
    print(json.dumps(preview, indent=2, sort_keys=True))

    if args.write:
        write_manifest(records, args.output_dir / "manifest_300clips.jsonl")
        write_json(split_summary, args.output_dir / "split_summary.json")
        write_json(storage_report, args.output_dir / "storage_report.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
