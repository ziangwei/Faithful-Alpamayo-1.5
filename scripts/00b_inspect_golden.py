#!/usr/bin/env python
"""Inspect vla_golden.parquet and (optionally) propose a reproducible clip subset.

CPU-only, read-only. This script never loads the Alpamayo model, never imports
torch/CUDA, and never touches PhysicalAI-AV media. It only reads the small golden
index parquet to answer, before any GPU run:

1. How many golden clips exist, and what columns / categorical values are there?
2. Which column holds the clip_id?
3. (optional) Draw a reproducible val subset of N clips.

Sampling is deliberately honest:
  - default ``uniform``: deterministic random sample (seeded). No cherry-picking.
  - ``--stratify-by COL``: proportional sample across a real categorical column,
    so the subset *covers* the scene-type distribution. Use this only on a column
    you saw in the inspection output, not on a guessed text field.

The stop/yield long-tail subset for the contribution should be derived POST-HOC
from the model's own parsed intents during evaluation, not pre-guessed here.

Safety:
  - Inspection-only by default. Selection runs only with ``--select N``.
  - Default output is a NEW file (``data/source_clip_ids_proposed.txt``); the
    script refuses to overwrite any existing file unless ``--force`` is passed,
    so a curated ``data/source_clip_ids.txt`` is never clobbered.

Examples:
    # Just look (no file written). Auto-finds the parquet in the HF dataset cache.
    python scripts/00b_inspect_golden.py

    # Draw a representative subset over a real low-cardinality column.
    # Pick N for the downstream use:
    #   * standalone reranker val/demo subset -> small N (e.g. 60)
    #   * source for 01_prepare_subset.py's 180/60/60 manifest -> N >= 300
    python scripts/00b_inspect_golden.py --select 300 --stratify-by scene_tags --write
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_FILENAME = "vla_golden.parquet"
DEFAULT_OUT = Path("data/source_clip_ids_proposed.txt")
MAX_CARDINALITY_FOR_VALUE_COUNTS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", type=Path, default=None, help="Explicit path to vla_golden.parquet.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Extra directory to search for the parquet.")
    parser.add_argument("--search-cache", action="store_true", help="Deep-scan a whole cache root if the dataset folder is not found (can be slow).")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset repo id (only with --allow-download).")
    parser.add_argument("--filename", default=DEFAULT_FILENAME, help="Parquet filename to look for / download.")
    parser.add_argument("--allow-download", action="store_true", help="Fetch the one parquet via huggingface_hub if not found locally.")
    parser.add_argument("--clip-id-column", default=None, help="Override clip-id column instead of auto-detecting.")
    parser.add_argument("--select", type=int, default=None, help="If set, propose this many clip ids. Omit to inspect only.")
    parser.add_argument("--stratify-by", default=None, help="Categorical column to sample proportionally across (implies stratified).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling.")
    parser.add_argument("--max-preview-rows", type=int, default=3, help="Sample rows to print.")
    parser.add_argument("--max-value-counts", type=int, default=20, help="Top values to print per low-cardinality column.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write selected clip ids.")
    parser.add_argument("--write", action="store_true", help="Actually write the clip-id file (requires --select).")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing --out file.")
    return parser.parse_args()


def require_pandas() -> Any:
    try:
        import pandas as pd  # noqa: WPS433 - intentional local import
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("pandas is required. Install with: pip install pandas pyarrow") from exc
    return pd


def search_roots(extra: Path | None) -> list[Path]:
    """Existing directories worth searching for the parquet, best first."""
    raw: list[Path] = []
    if extra:
        raw.append(extra)
    for env in (
        "PHYSICAL_AI_AV_LOCAL_DIR",
        "PHYSICAL_AI_AV_CACHE_DIR",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
    ):
        value = os.environ.get(env)
        if value:
            raw.append(Path(value).expanduser())
    raw.append(Path.home() / ".cache" / "huggingface")
    roots: list[Path] = []
    seen: set[str] = set()
    for root in raw:
        if str(root) in seen:
            continue
        seen.add(str(root))
        if root.exists():
            roots.append(root)
    return roots


def dataset_cache_dirname(repo_id: str) -> str:
    """HF cache folder name for a dataset repo, e.g. datasets--nvidia--PhysicalAI-..."""
    return "datasets--" + repo_id.replace("/", "--")


def _search_dataset_dirs(root: Path, slug: str, filename: str) -> Path | None:
    """Look only inside the dataset's own cache folder (fast), plus shallow hits at root."""
    for sub in (root / "hub" / slug, root / slug, root / "datasets" / slug):
        if sub.is_dir():
            for pattern in (filename, "*golden*.parquet"):
                for match in sub.rglob(pattern):
                    if match.is_file():
                        return match
    # PHYSICAL_AI_AV_LOCAL_DIR may point straight at materialized files.
    for pattern in (filename, "*golden*.parquet"):
        for match in root.glob(pattern):
            if match.is_file():
                return match
    return None


def find_parquet(args: argparse.Namespace) -> Path:
    """Locate the golden parquet: explicit path > targeted cache lookup > optional download.

    The default lookup only inspects the dataset's own cache folder, so it stays
    fast even on a large shared HF cache. Use --search-cache to deep-scan a whole
    cache root as a fallback.
    """
    if args.parquet is not None:
        if not args.parquet.exists():
            raise SystemExit(f"--parquet not found: {args.parquet}")
        return args.parquet

    slug = dataset_cache_dirname(args.repo_id)
    roots = search_roots(args.cache_dir)
    for root in roots:
        hit = _search_dataset_dirs(root, slug, args.filename)
        if hit is not None:
            return hit

    if args.search_cache:
        for root in roots:
            for pattern in (args.filename, "*golden*.parquet"):
                for match in root.rglob(pattern):
                    if match.is_file():
                        return match

    if args.allow_download:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("huggingface_hub is required for --allow-download.") from exc
        print(f"[info] downloading {args.filename} from {args.repo_id} (one small file) ...")
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            repo_type="dataset",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        return Path(downloaded)

    raise SystemExit(
        "Could not locate vla_golden.parquet. Pass --parquet PATH, set HF_HOME / "
        "PHYSICAL_AI_AV_CACHE_DIR to the dataset cache, add --search-cache for a "
        "deep scan, or use --allow-download."
    )


def pick_clip_id_column(df: Any, override: str | None) -> str:
    """Choose the clip-id column by exact name, then fuzzy 'clip'+'id', then 'clip'."""
    columns = list(df.columns)
    if override:
        if override not in columns:
            raise SystemExit(f"--clip-id-column '{override}' not in columns: {columns}")
        return override
    for name in columns:
        if name.lower() == "clip_id":
            return name
    for name in columns:
        low = name.lower()
        if "clip" in low and "id" in low:
            return name
    for name in columns:
        if "clip" in name.lower():
            return name
    raise SystemExit(f"No clip-id column found automatically. Use --clip-id-column. Columns: {columns}")


def low_cardinality_columns(df: Any, clip_col: str, max_card: int) -> list[tuple[str, int]]:
    """Columns (excluding clip id) with few unique values: inspection + stratify candidates."""
    out: list[tuple[str, int]] = []
    for name in df.columns:
        if name == clip_col:
            continue
        nunique = int(df[name].nunique(dropna=True))
        if 1 <= nunique <= max_card:
            out.append((name, nunique))
    return out


def truncate(value: Any, width: int = 80) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= width else text[: max(0, width - 3)] + "..."


def print_overview(df: Any, source: Path, clip_col: str, candidates: list[tuple[str, int]], args: argparse.Namespace) -> None:
    print("=" * 70)
    print(f"source        : {source}")
    print(f"rows          : {len(df)}")
    print(f"columns       : {len(df.columns)}")
    print(f"clip_id column: {clip_col}  (unique clips: {df[clip_col].nunique()})")
    print("-" * 70)
    print("schema (column : dtype : non-null):")
    for name in df.columns:
        print(f"  {name} : {df[name].dtype} : {int(df[name].notna().sum())}")

    print("-" * 70)
    if candidates:
        print(f"low-cardinality columns (good --stratify-by candidates, <= {MAX_CARDINALITY_FOR_VALUE_COUNTS} unique):")
        for name, nunique in candidates:
            print(f"  [{name}] unique={nunique}")
            counts = df[name].astype("string").value_counts(dropna=True).head(args.max_value_counts)
            for value, count in counts.items():
                print(f"      {count:>6}  {truncate(value, 60)}")
    else:
        print("no low-cardinality columns found; stratified sampling will be unavailable.")

    print("-" * 70)
    print(f"sample rows (first {args.max_preview_rows}):")
    for _, row in df.head(args.max_preview_rows).iterrows():
        print(f"  {clip_col}={truncate(row[clip_col], 40)}")
        for name in df.columns:
            if name == clip_col:
                continue
            print(f"      {name}: {truncate(row[name])}")
        print()


def sample_uniform(clip_ids: list[str], n: int, seed: int) -> list[str]:
    unique = list(dict.fromkeys(clip_ids))
    n = min(n, len(unique))
    pool = list(unique)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def sample_stratified(df: Any, clip_col: str, stratum_col: str, n: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    """Proportional (largest-remainder) sample across stratum_col, deterministic."""
    clip_to_stratum: dict[str, str] = {}
    for clip_id, value in zip(df[clip_col].astype(str), df[stratum_col].astype("string").fillna("<NA>")):
        if clip_id not in clip_to_stratum:
            clip_to_stratum[clip_id] = str(value)

    groups: dict[str, list[str]] = {}
    for clip_id, value in clip_to_stratum.items():
        groups.setdefault(value, []).append(clip_id)

    total = len(clip_to_stratum)
    n = min(n, total)

    raw = {key: len(members) / total * n for key, members in groups.items()}
    base = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = n - sum(base.values())
    by_fraction = sorted(groups.keys(), key=lambda key: (-(raw[key] - base[key]), key))
    index = 0
    while remainder > 0 and by_fraction:
        base[by_fraction[index % len(by_fraction)]] += 1
        remainder -= 1
        index += 1

    selected: list[str] = []
    allocation: dict[str, Any] = {}
    for key in sorted(groups.keys()):
        members = sorted(groups[key])
        random.Random(f"{seed}:{key}").shuffle(members)
        take = min(base.get(key, 0), len(members))
        selected.extend(members[:take])
        allocation[key] = {"available": len(members), "selected": take}

    if len(selected) < n:  # top up if rounding left a gap (group smaller than allocation)
        pool = [c for c in clip_to_stratum if c not in set(selected)]
        random.Random(seed).shuffle(pool)
        selected.extend(pool[: n - len(selected)])

    return selected, allocation


def write_clip_ids(clip_ids: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(clip_ids) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    pd = require_pandas()

    parquet_path = find_parquet(args)
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        raise SystemExit(f"{parquet_path} has 0 rows.")

    clip_col = pick_clip_id_column(df, args.clip_id_column)
    candidates = low_cardinality_columns(df, clip_col, MAX_CARDINALITY_FOR_VALUE_COUNTS)
    print_overview(df, parquet_path, clip_col, candidates, args)

    if args.select is None:
        print("=" * 70)
        print("[inspect-only] no --select given; nothing proposed or written.")
        if args.write:
            print("[note] --write ignored without --select.")
        return 0

    if args.stratify_by:
        if args.stratify_by not in df.columns:
            raise SystemExit(f"--stratify-by '{args.stratify_by}' not in columns: {list(df.columns)}")
        stratum_card = int(df[args.stratify_by].nunique(dropna=True))
        if stratum_card > MAX_CARDINALITY_FOR_VALUE_COUNTS and not args.force:
            raise SystemExit(
                f"--stratify-by '{args.stratify_by}' has {stratum_card} unique values "
                f"(> {MAX_CARDINALITY_FOR_VALUE_COUNTS}); this yields near-degenerate strata "
                "(~1 clip per value). Pick a low-cardinality column from the inspection "
                "output, or pass --force to override."
            )
        selected, allocation = sample_stratified(df, clip_col, args.stratify_by, args.select, args.seed)
        method = f"stratified by '{args.stratify_by}'"
    else:
        selected = sample_uniform(df[clip_col].astype(str).tolist(), args.select, args.seed)
        allocation = None
        method = "uniform random"

    print("=" * 70)
    summary: dict[str, Any] = {
        "method": method,
        "seed": args.seed,
        "requested": args.select,
        "num_total_clips": int(df[clip_col].nunique()),
        "num_selected": len(selected),
        "write_enabled": args.write,
        "out_path": str(args.out),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if allocation is not None:
        print("per-stratum allocation (value : available -> selected):")
        for key, info in sorted(allocation.items(), key=lambda kv: -kv[1]["selected"]):
            print(f"  {truncate(key, 40)} : {info['available']} -> {info['selected']}")

    preview = selected[:20]
    print(f"selected {len(selected)} clip ids (first {len(preview)} shown):")
    for clip_id in preview:
        print(f"  {clip_id}")

    if args.write:
        if args.out.exists() and not args.force:
            raise SystemExit(
                f"{args.out} already exists; refusing to overwrite. "
                "Use --force, or choose a different --out."
            )
        write_clip_ids(selected, args.out)
        print(f"[ok] wrote {len(selected)} clip ids -> {args.out}")
    else:
        print(f"[dry-run] would write {len(selected)} clip ids -> {args.out} (pass --write)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
