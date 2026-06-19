#!/usr/bin/env python
"""Build a reproducible val clip-id subset for the reranker from PhysicalAI-AV metadata.

CPU-only, read-only by default. Uses the OFFICIAL val split, keeps only clips that
are valid and carry the sensors Alpamayo needs (4 cameras + egomotion), then draws
N clips by proportional stratified sampling across driving conditions
(default: daypart x platform_class). Prints a coverage report. Writes the clip-id
list only with --write, and never overwrites an existing --out without --force.

The three metadata files are row-aligned (same length, same clip order):
  clip_index.parquet               : clip_is_valid, chunk, split          (RangeIndex)
  metadata/data_collection.parquet : country, month, hour_of_day, platform_class, radar_config (index=clip_id)
  metadata/feature_presence.parquet: per-sensor presence flags            (index=clip_id)

Example (run from repo root, after the parquets are in the HF cache):
  python scripts/01b_build_val_subset.py --select 60 --write
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from typing import Any

DATASET_SLUG = "datasets--nvidia--PhysicalAI-Autonomous-Vehicles"
DEFAULT_CAMERAS = ["camera_cross_left_120fov", "camera_front_wide_120fov",
                   "camera_cross_right_120fov", "camera_front_tele_30fov"]
DEFAULT_OUT = Path("data/source_clip_ids_proposed.txt")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot-dir", type=Path, default=None, help="Dir holding clip_index.parquet + metadata/. Auto-found under --cache-dir if omitted.")
    p.add_argument("--cache-dir", type=Path, default=Path("models/huggingface/hub"), help="HF cache root to auto-find the dataset snapshot.")
    p.add_argument("--split", default="val")
    p.add_argument("--select", type=int, default=60)
    p.add_argument("--stratify-by", default="daypart,platform_class", help="Comma list of metadata columns (daypart derived from hour_of_day).")
    p.add_argument("--required-cameras", default=",".join(DEFAULT_CAMERAS))
    p.add_argument("--no-require-egomotion", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--write", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def require_pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise SystemExit("pandas required: pip install pandas pyarrow") from e
    return pd


def find_snapshot(cache_dir: Path) -> Path | None:
    snaps = cache_dir / DATASET_SLUG / "snapshots"
    if snaps.is_dir():
        for snap in sorted(snaps.iterdir()):
            if (snap / "clip_index.parquet").is_file():
                return snap
    return None


def daypart(hour: int) -> str:
    if 6 <= hour <= 17:
        return "day"
    if hour in (5, 18, 19, 20):
        return "twilight"
    return "night"


def stratified_sample(clip_ids, strata, n, seed):
    groups: dict[str, list[str]] = {}
    for cid, s in zip(clip_ids, strata):
        groups.setdefault(s, []).append(cid)
    total = len(clip_ids); n = min(n, total)
    raw = {k: len(v) / total * n for k, v in groups.items()}
    base = {k: int(math.floor(x)) for k, x in raw.items()}
    rem = n - sum(base.values())
    order = sorted(groups, key=lambda k: (-(raw[k] - base[k]), str(k)))
    i = 0
    while rem > 0 and order:
        base[order[i % len(order)]] += 1; rem -= 1; i += 1
    selected: list[str] = []; alloc: dict[str, Any] = {}
    for k in sorted(groups, key=str):
        members = sorted(groups[k]); random.Random(f"{seed}:{k}").shuffle(members)
        take = min(base.get(k, 0), len(members)); selected += members[:take]
        alloc[str(k)] = {"available": len(members), "selected": take}
    if len(selected) < n:
        pool = [c for c in clip_ids if c not in set(selected)]
        random.Random(seed).shuffle(pool); selected += pool[: n - len(selected)]
    return selected, alloc


def main():
    a = parse_args(); pd = require_pandas()
    snap = a.snapshot_dir or find_snapshot(a.cache_dir)
    if snap is None:
        raise SystemExit(f"Could not find dataset snapshot under {a.cache_dir}/{DATASET_SLUG}/snapshots. Pass --snapshot-dir.")
    ci = pd.read_parquet(snap / "clip_index.parquet")
    dc = pd.read_parquet(snap / "metadata" / "data_collection.parquet")
    fp = pd.read_parquet(snap / "metadata" / "feature_presence.parquet")
    if not (len(ci) == len(dc) == len(fp)):
        raise SystemExit(f"row mismatch ci={len(ci)} dc={len(dc)} fp={len(fp)}; files not aligned.")

    df = dc.reset_index()
    id_col = df.columns[0]
    df["split"] = ci["split"].to_numpy()
    df["clip_is_valid"] = ci["clip_is_valid"].to_numpy()
    fp = fp.reindex(dc.index)  # align sensors by clip_id, order-independent
    cams = [c.strip() for c in a.required_cameras.split(",") if c.strip()]
    need = cams + ([] if a.no_require_egomotion else ["egomotion"])
    for col in need:
        if col not in fp.columns:
            raise SystemExit(f"feature_presence missing '{col}'. Has: {list(fp.columns)[:30]}")
        df[col] = fp[col].to_numpy()

    if "hour_of_day" in df.columns:
        df["daypart"] = df["hour_of_day"].astype(int).map(daypart)

    mask = (df["split"] == a.split) & (df["clip_is_valid"] == True)  # noqa: E712
    for col in need:
        mask = mask & (df[col] == True)  # noqa: E712
    pool = df[mask].copy()
    print(f"pool after filter (split={a.split}, valid, sensors={need}): {len(pool)} of {len(df)} clips")
    if len(pool) == 0:
        raise SystemExit("empty pool after filtering.")

    strat_cols = [c.strip() for c in a.stratify_by.split(",") if c.strip()]
    for c in strat_cols:
        if c not in pool.columns:
            raise SystemExit(f"stratify column '{c}' not found. Available: {list(pool.columns)}")
    strata = (pool[strat_cols].astype(str).agg(" | ".join, axis=1).tolist()
              if strat_cols else ["all"] * len(pool))
    clip_ids = pool[id_col].astype(str).tolist()
    selected, alloc = stratified_sample(clip_ids, strata, a.select, a.seed)

    print("=" * 70)
    print(json.dumps({"method": f"official {a.split} split -> sensor filter -> stratified by {strat_cols}",
                      "seed": a.seed, "requested": a.select, "selected": len(selected),
                      "out": str(a.out), "write": a.write}, indent=2))
    print("per-stratum (value : available -> selected):")
    for k, v in sorted(alloc.items(), key=lambda kv: -kv[1]["selected"]):
        print(f"  {k} : {v['available']} -> {v['selected']}")

    sel = pool[pool[id_col].astype(str).isin(set(selected))]
    print("coverage of selected set:")
    for axis in ["country", "daypart", "platform_class", "radar_config", "month"]:
        if axis in sel.columns:
            vc = sel[axis].value_counts()
            print(f"  {axis}: {dict(list(vc.items())[:8])}{' ...' if len(vc) > 8 else ''}")

    if a.write:
        if a.out.exists() and not a.force:
            raise SystemExit(f"{a.out} exists; use --force or a different --out.")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(selected) + "\n", encoding="utf-8")
        print(f"[ok] wrote {len(selected)} clip ids -> {a.out}")
    else:
        print(f"[dry-run] would write {len(selected)} clip ids -> {a.out} (pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
