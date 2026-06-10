#!/usr/bin/env python
"""Copy legacy/flat outputs into the canonical named-run directory layout.

This script never deletes source files. Without ``--execute`` it only prints
the migration plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.run_migration import MigrationSpec, build_migration_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy files and verify their hashes. Sources are never deleted.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_existing_source(spec: MigrationSpec) -> Path | None:
    return next((source for source in spec.sources if source.is_file()), None)


def process_spec(spec: MigrationSpec, execute: bool) -> dict[str, Any]:
    source = first_existing_source(spec)
    result: dict[str, Any] = {
        "artifact_type": spec.artifact_type,
        "source_candidates": [str(path) for path in spec.sources],
        "source": str(source) if source else None,
        "target": str(spec.target),
    }

    if spec.target.is_file():
        if source is None:
            result["status"] = "target_already_present"
            return result
        if sha256_file(source) == sha256_file(spec.target):
            result["status"] = "target_already_present_identical"
            return result
        result["status"] = "conflict_target_differs"
        return result

    if source is None:
        result["status"] = "missing_source"
        return result

    if not execute:
        result["status"] = "planned_copy"
        return result

    spec.target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, spec.target)
    if sha256_file(source) != sha256_file(spec.target):
        result["status"] = "copy_verification_failed"
        return result
    result["status"] = "copied_verified"
    return result


def main() -> int:
    args = parse_args()
    specs = build_migration_specs(run_name=args.run_name, split=args.split)
    results = [process_spec(spec, execute=args.execute) for spec in specs]
    statuses = [result["status"] for result in results]
    report = {
        "run_name": args.run_name,
        "split": args.split,
        "execute": args.execute,
        "source_deletion_performed": False,
        "results": results,
        "status_counts": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    failures = {
        "missing_source",
        "conflict_target_differs",
        "copy_verification_failed",
    }
    return 1 if any(status in failures for status in statuses) else 0


if __name__ == "__main__":
    raise SystemExit(main())
