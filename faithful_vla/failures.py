"""Failure mining helpers for baseline analysis outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any


DEFAULT_FAILURE_WEIGHTS = {
    "inconsistent": 100.0,
    "failed_check": 10.0,
    "fde": 1.0,
    "ade": 0.5,
}


def score_failure_row(
    row: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> float:
    """Score one metric row for qualitative review priority."""
    weights = weights or DEFAULT_FAILURE_WEIGHTS
    score = 0.0
    if not row.get("is_consistent", True):
        score += weights["inconsistent"]
    score += len(row.get("failed_checks", [])) * weights["failed_check"]
    score += float(row.get("fde_m", 0.0)) * weights["fde"]
    score += float(row.get("ade_m", 0.0)) * weights["ade"]
    return score


def rank_failure_rows(
    rows: list[dict[str, Any]],
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Return rows ranked by failure score, with rank and score added."""
    ranked: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["failure_score"] = score_failure_row(row, weights=weights)
        ranked.append(enriched)
    ranked.sort(key=lambda row: (-row["failure_score"], -float(row.get("fde_m", 0.0))))
    if top_k is not None:
        ranked = ranked[:top_k]
    for index, row in enumerate(ranked, start=1):
        row["failure_rank"] = index
    return ranked


def summarize_failures(
    rows: list[dict[str, Any]],
    top_rows: list[dict[str, Any]],
    run_name: str | None = None,
) -> dict[str, Any]:
    """Build a compact failure-mining summary."""
    failed_counter: Counter[str] = Counter()
    for row in rows:
        failed_counter.update(row.get("failed_checks", []))
    num_inconsistent = sum(1 for row in rows if not row.get("is_consistent", True))
    return {
        "run_name": run_name,
        "num_samples": len(rows),
        "num_inconsistent": num_inconsistent,
        "inconsistency_rate": num_inconsistent / len(rows) if rows else None,
        "failed_check_counts": dict(sorted(failed_counter.items())),
        "mean_ade_m": _mean(row.get("ade_m") for row in rows),
        "mean_fde_m": _mean(row.get("fde_m") for row in rows),
        "max_fde_m": max((float(row.get("fde_m", 0.0)) for row in rows), default=0.0),
        "top_failure_clip_ids": [row["clip_id"] for row in top_rows],
        "top_failure_sample_ids": [row.get("sample_id") for row in top_rows],
    }


def build_case_report_markdown(
    summary: dict[str, Any],
    top_rows: list[dict[str, Any]],
) -> str:
    """Build a Markdown report for top qualitative failure cases."""
    run_name = summary.get("run_name") or "unnamed"
    lines = [
        "# Failure Mining Report",
        "",
        f"- Run: `{run_name}`",
        f"- Samples: {summary.get('num_samples', 0)}",
        f"- Inconsistent samples: {summary.get('num_inconsistent', 0)}",
        f"- Inconsistency rate: {_format_percent(summary.get('inconsistency_rate'))}",
        f"- Failed check counts: `{summary.get('failed_check_counts', {})}`",
        "",
        "## Top Cases",
        "",
    ]
    for row in top_rows:
        lines.extend(_case_lines(row))
    return "\n".join(lines).rstrip() + "\n"


def _case_lines(row: dict[str, Any]) -> list[str]:
    cot = str(row.get("cot", "")).strip() or "(empty)"
    meta_action = str(row.get("meta_action", "")).strip() or "(empty)"
    return [
        f"### {row.get('failure_rank')}. `{row.get('clip_id')}`",
        "",
        f"- Sample: `{row.get('sample_id')}`",
        f"- Failure score: {float(row.get('failure_score', 0.0)):.3f}",
        f"- ADE/FDE: {float(row.get('ade_m', 0.0)):.3f} / {float(row.get('fde_m', 0.0)):.3f} m",
        f"- Consistent: {row.get('is_consistent')}",
        f"- Intents: `{row.get('intents', [])}`",
        f"- Failed checks: `{row.get('failed_checks', [])}`",
        f"- CoC: {cot}",
        f"- Meta action: {meta_action}",
        "",
    ]


def _mean(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"
