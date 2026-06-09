"""Shared output path helpers for named experiment runs."""

from __future__ import annotations

from pathlib import Path


def baseline_output_paths(split: str, run_name: str | None = None) -> dict[str, Path]:
    """Return baseline output paths for a split and optional run name."""
    if run_name:
        output_dir = Path("outputs/runs") / _clean_run_name(run_name) / "baseline"
    else:
        output_dir = Path("outputs/baseline")
    return {
        "predictions": output_dir / f"{split}_predictions.jsonl",
        "trajectories": output_dir / f"{split}_trajectories.npz",
        "runtime": output_dir / f"{split}_runtime.jsonl",
    }


def metrics_output_paths(split: str = "val", run_name: str | None = None) -> dict[str, Path]:
    """Return metrics input and output paths for a split and optional run name."""
    baseline_paths = baseline_output_paths(split=split, run_name=run_name)
    if run_name:
        output_dir = Path("outputs/runs") / _clean_run_name(run_name) / "metrics"
    else:
        output_dir = Path("outputs/metrics")
    return {
        "predictions": baseline_paths["predictions"],
        "trajectories": baseline_paths["trajectories"],
        "summary": output_dir / "summary.json",
        "per_sample": output_dir / "per_sample_metrics.jsonl",
        "inconsistency": output_dir / "inconsistency_examples.jsonl",
    }


def analysis_output_paths(run_name: str | None = None) -> dict[str, Path]:
    """Return failure-analysis output paths for an optional run name."""
    if run_name:
        output_dir = Path("outputs/runs") / _clean_run_name(run_name) / "analysis"
    else:
        output_dir = Path("outputs/analysis")
    return {
        "failure_summary": output_dir / "failure_summary.json",
        "top_failures": output_dir / "top_failures.jsonl",
        "case_report": output_dir / "case_report.md",
    }


def _clean_run_name(run_name: str) -> str:
    cleaned = run_name.strip()
    if not cleaned:
        raise ValueError("run_name must be non-empty")
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned or ".." in cleaned:
        raise ValueError("run_name must be a simple directory name, not a path")
    return cleaned
