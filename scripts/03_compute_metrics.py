#!/usr/bin/env python
"""Compute metrics from saved Alpamayo baseline outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.metrics import (
    DEFAULT_CONSISTENCY_THRESHOLDS,
    DEFAULT_INTENT_KEYWORDS,
    check_reasoning_action_consistency,
    compute_trajectory_metrics,
    parse_intents,
    summarize_metric_rows,
)
from faithful_vla.run_paths import metrics_output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Read/write default files under outputs/runs/<run-name>/.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=None,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/metrics.yaml"))
    parser.add_argument("--time-step", type=float, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--per-sample-jsonl",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--inconsistency-jsonl",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for key in ("clip_id", "sample_id", "pred_xyz_npz_key", "gt_xyz_npz_key"):
            if key not in row:
                raise ValueError(f"{path}:{line_number} missing required field: {key}")
        rows.append(row)
    return rows


def load_metrics_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError:
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded or {}


def resolve_thresholds(config: dict[str, Any]) -> dict[str, float]:
    thresholds = dict(DEFAULT_CONSISTENCY_THRESHOLDS)
    thresholds.update(config.get("consistency_thresholds", {}) or {})
    return {key: float(value) for key, value in thresholds.items()}


def resolve_intent_keywords(config: dict[str, Any]) -> dict[str, list[str]]:
    configured = ((config.get("intent_parser", {}) or {}).get("intents", {}) or {})
    if not configured:
        return DEFAULT_INTENT_KEYWORDS
    return {key: list(value) for key, value in configured.items()}


def resolve_time_step(args: argparse.Namespace, config: dict[str, Any]) -> float:
    if args.time_step is not None:
        return args.time_step
    trajectory_config = config.get("trajectory", {}) or {}
    return float(trajectory_config.get("time_step", 0.1))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def build_metric_row(
    prediction: dict[str, Any],
    pred_xyz: Any,
    gt_xyz: Any,
    time_step: float,
    intent_keywords: dict[str, list[str]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    trajectory_metrics = compute_trajectory_metrics(
        pred_xyz.tolist(),
        gt_xyz.tolist(),
        time_step=time_step,
    )
    intents = parse_intents(
        {
            "cot": prediction.get("cot"),
            "meta_action": prediction.get("meta_action"),
            "answer": prediction.get("answer"),
        },
        intent_keywords=intent_keywords,
    )
    consistency = check_reasoning_action_consistency(
        intents=intents,
        behavior=trajectory_metrics,
        thresholds=thresholds,
    )
    return {
        "clip_id": prediction["clip_id"],
        "sample_id": prediction["sample_id"],
        "trajectory_sample_id": prediction.get("trajectory_sample_id", 0),
        "split": prediction.get("split"),
        "t0_us": prediction.get("t0_us"),
        "cot": prediction.get("cot", ""),
        "meta_action": prediction.get("meta_action", ""),
        "intents": intents,
        **trajectory_metrics,
        **consistency,
    }


def main() -> int:
    args = parse_args()
    import numpy as np

    default_paths = metrics_output_paths(split=args.split, run_name=args.run_name)
    predictions_path = args.predictions or default_paths["predictions"]
    trajectories_path = args.trajectories or default_paths["trajectories"]
    summary_json = args.summary_json or default_paths["summary"]
    per_sample_jsonl = args.per_sample_jsonl or default_paths["per_sample"]
    inconsistency_jsonl = args.inconsistency_jsonl or default_paths["inconsistency"]

    config = load_metrics_config(args.config)
    thresholds = resolve_thresholds(config)
    intent_keywords = resolve_intent_keywords(config)
    time_step = resolve_time_step(args, config)

    predictions = read_jsonl(predictions_path)
    trajectories = np.load(trajectories_path)

    metric_rows: list[dict[str, Any]] = []
    missing_keys: list[dict[str, str]] = []
    for prediction in predictions:
        pred_key = prediction["pred_xyz_npz_key"]
        gt_key = prediction["gt_xyz_npz_key"]
        if pred_key not in trajectories or gt_key not in trajectories:
            missing_keys.append(
                {
                    "clip_id": prediction["clip_id"],
                    "sample_id": prediction["sample_id"],
                    "pred_xyz_npz_key": pred_key,
                    "gt_xyz_npz_key": gt_key,
                }
            )
            continue
        metric_rows.append(
            build_metric_row(
                prediction=prediction,
                pred_xyz=trajectories[pred_key],
                gt_xyz=trajectories[gt_key],
                time_step=time_step,
                intent_keywords=intent_keywords,
                thresholds=thresholds,
            )
        )

    summary = summarize_metric_rows(metric_rows)
    summary.update(
        {
            "predictions_jsonl": str(predictions_path),
            "trajectories_npz": str(trajectories_path),
            "per_sample_jsonl": str(per_sample_jsonl),
            "inconsistency_examples_jsonl": str(inconsistency_jsonl),
            "config": str(args.config),
            "run_name": args.run_name,
            "split": args.split,
            "time_step": time_step,
            "num_prediction_rows": len(predictions),
            "num_missing_trajectory_rows": len(missing_keys),
            "missing_trajectory_rows": missing_keys[:20],
            "thresholds": thresholds,
        }
    )

    inconsistent_rows = [row for row in metric_rows if not row["is_consistent"]]
    write_json(summary_json, summary)
    write_jsonl(per_sample_jsonl, metric_rows)
    write_jsonl(inconsistency_jsonl, inconsistent_rows)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary: {summary_json}")
    print(f"Wrote per-sample metrics: {per_sample_jsonl}")
    print(f"Wrote inconsistency examples: {inconsistency_jsonl}")
    return 1 if missing_keys else 0


if __name__ == "__main__":
    raise SystemExit(main())
