#!/usr/bin/env python
"""k-of-N scaling: how do oracle best-of-k and MBR/consensus-of-k grow with k?

CPU-only, read-only over an existing baseline run; no GPU needed.
Candidates are i.i.d. diffusion samples, so averaging over all C(N,k) candidate
subsets is an unbiased estimate of "what if we had only drawn k samples":
  - oracle-of-k : min-ADE candidate in the subset (selection upper bound at budget k)
  - MBR-of-k    : candidate nearest the subset centroid (same rule as 04_rerank)
Sanity anchors: k=1 equals the random-candidate baseline for both curves;
k=N reproduces the headline oracle (03c) and consensus (04) numbers.

Per-k extras (answer "how big should N be"): oracle saturation vs the full-N ceiling
and the marginal ADE won by the k-th sample (diminishing returns). Optional --plot
writes the scaling curve.

Reads : outputs/runs/<run>/baseline/{<split>_predictions.jsonl, <split>_trajectories.npz}
Writes: outputs/runs/<run>/analysis/k_scaling.json  (+ k_scaling.png with --plot; new files only)

Run:
  python scripts/03d_k_scaling.py --run-name val_cand5_n1000 --plot
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
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
    p.add_argument("--bootstrap", type=int, default=2000, help="paired bootstrap resamples (0 = skip CIs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--plot", action="store_true", help="also write k_scaling.png (scaling curve) next to the JSON")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def mbr_index(arrs, idxs) -> int:
    """04_rerank's consensus rule restricted to a subset: candidate (by global index)
    nearest the subset centroid, distance = mean per-timestep L2 on xy."""
    import numpy as np
    centroid = np.mean([arrs[i] for i in idxs], axis=0)
    return min(idxs, key=lambda i: float(np.mean(np.linalg.norm(arrs[i] - centroid, axis=1))))


def per_clip_curves(cands: list[dict[str, float]], arrs) -> dict[int, dict[str, float]]:
    """{k: mean over all C(N,k) subsets of oracle/MBR ADE+FDE} for one clip.

    cands: [{"ade":..,"fde":..}] in candidate order; arrs: matching (T,2) xy arrays.
    """
    n = len(cands)
    out: dict[int, dict[str, float]] = {}
    for k in range(1, n + 1):
        acc = {"oracle_ade": 0.0, "oracle_fde": 0.0, "mbr_ade": 0.0, "mbr_fde": 0.0}
        count = 0
        for idxs in itertools.combinations(range(n), k):
            best = min(idxs, key=lambda i: cands[i]["ade"])
            sel = mbr_index(arrs, idxs)
            acc["oracle_ade"] += cands[best]["ade"]
            acc["oracle_fde"] += cands[best]["fde"]
            acc["mbr_ade"] += cands[sel]["ade"]
            acc["mbr_fde"] += cands[sel]["fde"]
            count += 1
        out[k] = {key: value / count for key, value in acc.items()}
    return out


def bootstrap_ci(diffs, n_boot: int, seed: int) -> dict[str, Any]:
    """Paired bootstrap on mean(diffs); positive = MBR-of-k better than first-sample."""
    import numpy as np
    d = np.asarray(diffs, dtype=float)
    if n_boot <= 0 or d.size == 0:
        return {"mean_improve_m": round(float(d.mean()), 4) if d.size else None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return {"mean_improve_m": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
            "frac_better": round(float((means > 0).mean()), 3)}


def analyze(clips: list[dict[str, Any]], n_cand: int, n_boot: int, seed: int) -> dict[str, Any]:
    """Aggregate per-clip curves into per-k means + gap-closed% + CIs vs first.

    Per-k extras that directly answer "how big should N be":
      - oracle_sat_pct    : fraction of the *full-N* oracle ceiling already reached
                            at budget k = (first - oracle_k)/(first - oracle_N).
      - oracle_marginal_ade / mbr_marginal_ade : ADE won by adding the k-th sample
                            (oracle_{k-1} - oracle_k). k=1 is None (no k=0). This is
                            the diminishing-returns signal: pick N where it flattens.
    """
    first_ade, first_fde = [c["first_ade"] for c in clips], [c["first_fde"] for c in clips]
    fa, ff = _mean(first_ade), _mean(first_fde)
    oracle_n = _mean([c["curves"][n_cand]["oracle_ade"] for c in clips])
    # per-k means first, so we can compute marginal (k-1 -> k) improvements.
    oa_k = {k: _mean([c["curves"][k]["oracle_ade"] for c in clips]) for k in range(1, n_cand + 1)}
    ma_k = {k: _mean([c["curves"][k]["mbr_ade"] for c in clips]) for k in range(1, n_cand + 1)}
    rows = []
    for k in range(1, n_cand + 1):
        oa, ma = oa_k[k], ma_k[k]
        of = _mean([c["curves"][k]["oracle_fde"] for c in clips])
        mf = _mean([c["curves"][k]["mbr_fde"] for c in clips])
        gap = (fa - ma) / (fa - oracle_n) * 100 if fa > oracle_n else None
        sat = (fa - oa) / (fa - oracle_n) * 100 if fa > oracle_n else None
        o_marg = (oa_k[k - 1] - oa) if k > 1 else None
        m_marg = (ma_k[k - 1] - ma) if k > 1 else None
        ci = bootstrap_ci([c["first_ade"] - c["curves"][k]["mbr_ade"] for c in clips], n_boot, seed)
        rows.append({"k": k, "oracle_ade": round(oa, 4), "oracle_fde": round(of, 4),
                     "mbr_ade": round(ma, 4), "mbr_fde": round(mf, 4),
                     "mbr_gap_closed_ade_pct": round(gap, 2) if gap is not None else None,
                     "oracle_sat_pct": round(sat, 2) if sat is not None else None,
                     "oracle_marginal_ade": round(o_marg, 4) if o_marg is not None else None,
                     "mbr_marginal_ade": round(m_marg, 4) if m_marg is not None else None,
                     "mbr_vs_first_ade": ci})
    return {"num_clips": len(clips),
            "first_sample_ade": round(fa, 4), "first_sample_fde": round(ff, 4),
            "oracle_full_n_ade": round(oracle_n, 4), "per_k": rows}


def print_table(tag: str, block: dict[str, Any]) -> None:
    print(f"\n[{tag}] n={block['num_clips']}  first ADE {block['first_sample_ade']}"
          f"  oracle-of-N {block['oracle_full_n_ade']}  (gap%/sat% relative to oracle-of-N)")
    print(f"{'k':>2}  {'oracle':>7}  {'o_sat%':>6}  {'o_marg':>7}  {'MBR':>7}  {'gap%':>6}  "
          f"{'m_marg':>7}  {'MBR vs first (CI95)':>24}")
    for r in block["per_k"]:
        ci = r["mbr_vs_first_ade"].get("ci95")
        ci_s = f"{r['mbr_vs_first_ade']['mean_improve_m']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "-"
        gap = f"{r['mbr_gap_closed_ade_pct']:.1f}" if r["mbr_gap_closed_ade_pct"] is not None else "-"
        sat = f"{r['oracle_sat_pct']:.1f}" if r["oracle_sat_pct"] is not None else "-"
        om = f"{r['oracle_marginal_ade']:+.3f}" if r["oracle_marginal_ade"] is not None else "-"
        mm = f"{r['mbr_marginal_ade']:+.3f}" if r["mbr_marginal_ade"] is not None else "-"
        print(f"{r['k']:>2}  {r['oracle_ade']:>7.4f}  {sat:>6}  {om:>7}  {r['mbr_ade']:>7.4f}  "
              f"{gap:>6}  {mm:>7}  {ci_s:>24}")


def make_plot(report: dict[str, Any], path: Path) -> bool:
    """Draw oracle-of-k and MBR-of-k ADE vs sample budget k (with MBR CI band).
    Returns False (no-op) if matplotlib is unavailable. Writes a new PNG only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    blocks = [("overall", report["overall"])]
    if report.get("stop_yield_subset", {}).get("num_clips"):
        blocks.append(("stop/yield", report["stop_yield_subset"]))
    fig, axes = plt.subplots(1, len(blocks), figsize=(5.4 * len(blocks), 4.4), squeeze=False)
    for ax, (tag, blk) in zip(axes[0], blocks):
        ks = [r["k"] for r in blk["per_k"]]
        oracle = [r["oracle_ade"] for r in blk["per_k"]]
        mbr = [r["mbr_ade"] for r in blk["per_k"]]
        # CI band on MBR: mbr_ade +/- half the vs-first CI width (visual guide only)
        band = []
        for r in blk["per_k"]:
            ci = r["mbr_vs_first_ade"].get("ci95")
            band.append(((ci[1] - ci[0]) / 2.0) if ci else 0.0)
        mbr_lo = [m - b for m, b in zip(mbr, band)]
        mbr_hi = [m + b for m, b in zip(mbr, band)]
        ax.axhline(blk["first_sample_ade"], ls="--", lw=1, color="#888", label="first-sample (deployed)")
        ax.plot(ks, oracle, "o-", color="#2a7", label="oracle best-of-k (ceiling)")
        ax.plot(ks, mbr, "s-", color="#27a", label="MBR / consensus-of-k")
        ax.fill_between(ks, mbr_lo, mbr_hi, color="#27a", alpha=0.15)
        ax.set_xticks(ks)
        ax.set_xlabel("sample budget k")
        ax.set_ylabel("ADE (m)")
        ax.set_title(f"{tag}  (n={blk['num_clips']})")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("k-of-N scaling: selection quality vs number of sampled trajectories")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def build_report(preds: list[dict[str, Any]], traj, *, time_step: float = 0.1,
                 stop_yield_intents: str = "stop,yield,slow_down",
                 bootstrap: int = 2000, seed: int = 0,
                 run_name: str | None = None, split: str = "val") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure (no file IO) core: preds + trajectory mapping -> (report, clips).

    `traj` is any mapping key -> (T, >=2) array (an npz handle or a plain dict),
    so this is unit-testable without touching disk. Mirrors scripts/07's build_report.
    """
    import numpy as np
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        pk, gk = p.get("pred_xyz_npz_key"), p.get("gt_xyz_npz_key")
        if pk not in traj or gk not in traj:
            continue
        m = compute_trajectory_metrics(np.asarray(traj[pk]).tolist(), np.asarray(traj[gk]).tolist(),
                                       time_step=time_step)
        groups.setdefault(p["sample_id"], []).append({
            "tid": p.get("trajectory_sample_id", 0), "ade": m["ade_m"], "fde": m["fde_m"],
            "xy": np.asarray(traj[pk], dtype=float)[:, :2],
            "cot": p.get("cot"), "meta_action": p.get("meta_action"), "answer": p.get("answer"),
        })

    n_cand = max((len(v) for v in groups.values()), default=0)
    sy_intents = set(s.strip() for s in stop_yield_intents.split(",") if s.strip())
    clips, dropped = [], 0
    for sid, cands in groups.items():
        if len(cands) != n_cand:
            dropped += 1
            continue
        cands.sort(key=lambda c: c["tid"])
        first = next((c for c in cands if c["tid"] == 0), cands[0])
        intents = set(parse_intents({"cot": first["cot"], "meta_action": first["meta_action"],
                                     "answer": first["answer"]}))
        clips.append({
            "first_ade": first["ade"], "first_fde": first["fde"], "intents": intents,
            "curves": per_clip_curves([{"ade": c["ade"], "fde": c["fde"]} for c in cands],
                                      [c["xy"] for c in cands]),
        })

    overall = analyze(clips, n_cand, bootstrap, seed)
    sy = [c for c in clips if c["intents"] & sy_intents]
    stop_yield = analyze(sy, n_cand, bootstrap, seed) if sy else {"num_clips": 0}

    report = {"run_name": run_name, "split": split, "num_candidates": n_cand,
              "num_clips": len(clips), "dropped_incomplete_clips": dropped,
              "bootstrap": bootstrap, "seed": seed,
              "random_candidate_ade": round(_mean(
                  [c["curves"][1]["mbr_ade"] for c in clips]), 4) if clips else None,
              "overall": overall, "stop_yield_subset": stop_yield}
    return report, clips


def main() -> int:
    a = parse_args()
    import numpy as np
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    preds = read_jsonl(a.predictions or bp["predictions"])
    traj = np.load(a.trajectories or bp["trajectories"])

    report, _clips = build_report(
        preds, traj, time_step=a.time_step, stop_yield_intents=a.stop_yield_intents,
        bootstrap=a.bootstrap, seed=a.seed, run_name=a.run_name, split=a.split)
    print_table("overall", report["overall"])
    if report["stop_yield_subset"].get("num_clips"):
        print_table("stop_yield", report["stop_yield_subset"])

    out = a.out
    if out is None and a.run_name:
        out = Path("outputs/runs") / a.run_name / "analysis" / "k_scaling.json"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\n[ok] wrote {out}")
    if a.plot:
        png = (out.with_name("k_scaling.png") if out else Path("k_scaling.png"))
        if make_plot(report, png):
            print(f"[ok] wrote {png}")
        else:
            print("[warn] matplotlib unavailable -> skipped plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
