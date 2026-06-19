#!/usr/bin/env python
"""Gate B: oracle best-of-N vs first-sample trajectory selection gap.

Self-contained, CPU-only, read-only. Reads the baseline predictions JSONL +
trajectories NPZ (one row per clip x candidate), computes ADE/FDE of every
candidate vs GT, then per clip compares:
  - first-sample     : default candidate 0 (what Alpamayo outputs today)
  - oracle best-of-N : the candidate with the lowest ADE (the upper bound)
  - random candidate : mean over candidates (sanity, ~= first if no selection signal)
Reports the oracle gap (recoverable selection error) overall + on the stop/yield
subset, plus candidate diversity. This is the go/no-go for the reranker.

Run (after the baseline run produced outputs/runs/<run>/baseline/):
  python scripts/03c_oracle_gap.py --run-name val_cand5
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.metrics import compute_trajectory_metrics, parse_intents
from faithful_vla.run_paths import baseline_output_paths


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--run-name", default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--stop-yield-intents", default="stop,yield,slow_down")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def analyze(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    first_ade, first_fde, oracle_ade, oracle_fde, rand_ade = [], [], [], [], []
    improved, ade_std, ade_range = 0, [], []
    for rows in groups.values():
        ades = [r["ade_m"] for r in rows]
        first = next((r for r in rows if r.get("trajectory_sample_id", 0) == 0),
                     sorted(rows, key=lambda r: r.get("trajectory_sample_id", 0))[0])
        best = min(rows, key=lambda r: r["ade_m"])
        first_ade.append(first["ade_m"]); first_fde.append(first["fde_m"])
        oracle_ade.append(best["ade_m"]); oracle_fde.append(best["fde_m"])
        rand_ade.append(_mean(ades))
        if best["ade_m"] < first["ade_m"] - 1e-9:
            improved += 1
        if len(ades) > 1:
            ade_std.append(statistics.pstdev(ades)); ade_range.append(max(ades) - min(ades))
    n = len(groups)
    fa, oa = _mean(first_ade), _mean(oracle_ade)
    ff, of = _mean(first_fde), _mean(oracle_fde)
    return {
        "num_clips": n,
        "first_sample_ade": round(fa, 4), "oracle_ade": round(oa, 4),
        "random_candidate_ade": round(_mean(rand_ade), 4),
        "first_sample_fde": round(ff, 4), "oracle_fde": round(of, 4),
        "ade_oracle_gap_abs": round(fa - oa, 4),
        "ade_oracle_gap_pct": round((fa - oa) / fa * 100, 2) if fa else None,
        "fde_oracle_gap_abs": round(ff - of, 4),
        "fde_oracle_gap_pct": round((ff - of) / ff * 100, 2) if ff else None,
        "clips_with_better_candidate": improved,
        "frac_clips_improvable": round(improved / n, 3) if n else None,
        "mean_within_clip_ade_std": round(_mean(ade_std), 4) if ade_std else None,
        "mean_within_clip_ade_range": round(_mean(ade_range), 4) if ade_range else None,
    }


def main() -> int:
    a = parse_args()
    import numpy as np
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    pred_path = a.predictions or bp["predictions"]
    traj_path = a.trajectories or bp["trajectories"]
    preds = read_jsonl(pred_path)
    traj = np.load(traj_path)

    sy = set(s.strip() for s in a.stop_yield_intents.split(",") if s.strip())
    groups: dict[str, list[dict[str, Any]]] = {}
    missing = 0
    for p in preds:
        pk, gk = p.get("pred_xyz_npz_key"), p.get("gt_xyz_npz_key")
        if pk not in traj or gk not in traj:
            missing += 1; continue
        m = compute_trajectory_metrics(traj[pk].tolist(), traj[gk].tolist(), time_step=a.time_step)
        intents = parse_intents({"cot": p.get("cot"), "meta_action": p.get("meta_action"), "answer": p.get("answer")})
        row = {"ade_m": m["ade_m"], "fde_m": m["fde_m"],
               "trajectory_sample_id": p.get("trajectory_sample_id", 0), "intents": intents}
        groups.setdefault(p["sample_id"], []).append(row)

    overall = analyze(groups)
    sy_groups = {sid: rows for sid, rows in groups.items()
                 if any(i in sy for r in rows if r.get("trajectory_sample_id", 0) == 0 for i in r["intents"])}
    stop_yield = analyze(sy_groups) if sy_groups else {"num_clips": 0}

    report = {"run_name": a.run_name, "split": a.split, "num_prediction_rows": len(preds),
              "num_missing_trajectory_rows": missing,
              "candidates_per_clip": round(_mean([len(v) for v in groups.values()]), 2) if groups else 0,
              "overall": overall, "stop_yield_subset": stop_yield}
    print(json.dumps(report, indent=2))

    out = a.out
    if out is None and a.run_name:
        out = Path("outputs/runs") / a.run_name / "analysis" / "oracle_gap.json"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
