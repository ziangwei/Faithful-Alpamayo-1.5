"""Migration definitions for legacy and flat experiment output layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from faithful_vla.run_paths import baseline_output_paths, metrics_output_paths


@dataclass(frozen=True)
class MigrationSpec:
    """Candidate sources and canonical target for one output artifact."""

    artifact_type: str
    sources: tuple[Path, ...]
    target: Path


def build_migration_specs(run_name: str, split: str = "val") -> list[MigrationSpec]:
    """Build copy specifications from flat/legacy layouts to the canonical run layout."""
    run_root = Path("outputs/runs") / run_name
    baseline_paths = baseline_output_paths(split=split, run_name=run_name)
    metrics_paths = metrics_output_paths(split=split, run_name=run_name)

    baseline_names = {
        "predictions": f"{split}_predictions.jsonl",
        "runtime": f"{split}_runtime.jsonl",
        "trajectories": f"{split}_trajectories.npz",
    }
    specs = [
        MigrationSpec(
            artifact_type=f"baseline_{key}",
            sources=(
                run_root / filename,
                Path("outputs/baseline") / filename,
            ),
            target=baseline_paths[key],
        )
        for key, filename in baseline_names.items()
    ]

    metric_names = {
        "summary": "summary.json",
        "per_sample": "per_sample_metrics.jsonl",
        "inconsistency": "inconsistency_examples.jsonl",
    }
    specs.extend(
        MigrationSpec(
            artifact_type=f"metrics_{key}",
            sources=(
                run_root / filename,
                Path("outputs/metrics") / filename,
            ),
            target=metrics_paths[key],
        )
        for key, filename in metric_names.items()
    )
    return specs
