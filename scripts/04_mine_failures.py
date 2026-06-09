#!/usr/bin/env python
"""Mine and summarize qualitative failure cases from metric outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.failures import (
    build_case_report_markdown,
    rank_failure_rows,
    summarize_failures,
)
from faithful_vla.run_paths import analysis_output_paths, metrics_output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Read metrics and write analysis under outputs/runs/<run-name>/.",
    )
    parser.add_argument("--per-sample-jsonl", type=Path, default=None)
    parser.add_argument("--failure-summary-json", type=Path, default=None)
    parser.add_argument("--top-failures-jsonl", type=Path, default=None)
    parser.add_argument("--case-report-md", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for key in ("clip_id", "sample_id", "ade_m", "fde_m", "is_consistent"):
            if key not in row:
                raise ValueError(f"{path}:{line_number} missing required field: {key}")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    metric_paths = metrics_output_paths(split=args.split, run_name=args.run_name)
    analysis_paths = analysis_output_paths(run_name=args.run_name)
    per_sample_jsonl = args.per_sample_jsonl or metric_paths["per_sample"]
    failure_summary_json = args.failure_summary_json or analysis_paths["failure_summary"]
    top_failures_jsonl = args.top_failures_jsonl or analysis_paths["top_failures"]
    case_report_md = args.case_report_md or analysis_paths["case_report"]

    rows = read_jsonl(per_sample_jsonl)
    top_rows = rank_failure_rows(rows, top_k=args.top_k)
    summary = summarize_failures(rows, top_rows=top_rows, run_name=args.run_name)
    summary.update(
        {
            "split": args.split,
            "per_sample_jsonl": str(per_sample_jsonl),
            "failure_summary_json": str(failure_summary_json),
            "top_failures_jsonl": str(top_failures_jsonl),
            "case_report_md": str(case_report_md),
            "top_k": args.top_k,
        }
    )
    report = build_case_report_markdown(summary, top_rows)

    write_json(failure_summary_json, summary)
    write_jsonl(top_failures_jsonl, top_rows)
    case_report_md.parent.mkdir(parents=True, exist_ok=True)
    case_report_md.write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote failure summary: {failure_summary_json}")
    print(f"Wrote top failures: {top_failures_jsonl}")
    print(f"Wrote case report: {case_report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
