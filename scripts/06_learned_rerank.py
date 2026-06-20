#!/usr/bin/env python
"""Tier-2 learned trajectory reranker (leave-one-clip-out CV), numpy-only.

The question this answers
-------------------------
A free, training-free MBR/consensus selector already recovers ~26% of the oracle
gap (see scripts/04_rerank.py). Tier 2 asks the obvious follow-up:

    Can a *learned* scorer, trained on the same cheap features
    (dynamics + distance-to-consensus), beat free MBR?

Honest prior (stated up front, not after seeing the result): the reasoning-aware
vs reasoning-blind ablation already showed cheap post-hoc features carry little
*extra* selection signal beyond plain centrality. So we expect learned ~= MBR;
genuinely beating MBR would require richer (visual / model-internal) features.
Either outcome is a clean conclusion -- "learned == MBR" means MBR is at the
ceiling of cheap features, which is itself worth stating.

Design
------
- Inputs: the existing baseline run (predictions JSONL + trajectories NPZ, one
  row per clip x candidate). No GPU, no new data, read-only.
- Features per candidate (within-clip z-scored): final_speed, speed_delta,
  max_abs_jerk, heading_change, lateral_disp (reused verbatim from 04's
  candidate_features) + dist_to_consensus (mean L2 of the candidate path to the
  clip centroid -- the exact signal MBR exploits, given to the model so it *can*
  match MBR).
- Labels: per clip the min-ADE candidate is the positive (is_best) for the
  logistic model; the ridge model regresses within-clip z-scored ADE. Selection
  never uses GT.
- Leave-one-clip-out CV: for each clip i, fit on the other N-1 clips, score clip
  i's candidates, pick argmax P(best) (logreg) / argmin predicted ADE (ridge).
  This is honest held-out evaluation that reuses all N clips -- no separate test
  split is needed or wasted.
- Models are numpy-only (ridge closed-form; logistic regression by full-batch GD)
  to keep the project's light dependency footprint (sklearn is not a dependency).

Evaluation (GT only scores, never selects): mean ADE/FDE, oracle gap closed, and
paired bootstrap CIs both vs first-sample (identical method to 04) and **vs MBR**
(the contrast that actually matters), overall and on the stop/yield subset, with
per-case win/loss. Writes a JSON report + per-clip selection JSONL.

Run (after the baseline run produced outputs/runs/<run>/baseline/):
  python scripts/06_learned_rerank.py --run-name val_cand5_n1000
"""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faithful_vla.metrics import compute_trajectory_metrics, parse_intents
from faithful_vla.run_paths import baseline_output_paths

STOP_LIKE = {"stop", "yield", "slow_down"}
FEAT_KEYS = ("final_speed", "speed_delta", "max_abs_jerk", "heading_change", "lateral_disp")


# Self-contained copies of the proven helpers from scripts/04_rerank.py (kept byte-for-byte
# identical on purpose). 06 stays runnable without importing a digit-prefixed sibling, matching
# the repo pattern where each script re-defines its small helpers and imports only faithful_vla.
def candidate_features(xyz: list[list[float]], dt: float) -> dict[str, float]:
    pts = [(float(p[0]), float(p[1])) for p in xyz]
    if len(pts) < 2:
        return dict(final_speed=0.0, speed_delta=0.0, max_abs_jerk=0.0, heading_change=0.0, lateral_disp=0.0)
    speeds = [math.dist(pts[i], pts[i - 1]) / dt for i in range(1, len(pts))]
    final_speed = speeds[-1]
    speed_delta = speeds[0] - speeds[-1]                       # >0 = decelerating
    accels = [(speeds[i] - speeds[i - 1]) / dt for i in range(1, len(speeds))]
    jerks = [(accels[i] - accels[i - 1]) / dt for i in range(1, len(accels))]
    max_abs_jerk = max((abs(j) for j in jerks), default=0.0)
    h0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    h1 = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
    heading_change = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
    lateral = 0.0
    for p in pts:
        dx, dy = p[0] - pts[0][0], p[1] - pts[0][1]
        lateral = max(lateral, abs(-math.sin(h0) * dx + math.cos(h0) * dy))
    return dict(final_speed=final_speed, speed_delta=speed_delta, max_abs_jerk=max_abs_jerk,
                heading_change=heading_change, lateral_disp=lateral)


def zscore(values: list[float]) -> list[float]:
    m = sum(values) / len(values) if values else 0.0
    s = statistics.pstdev(values) if len(values) > 1 else 0.0
    return [((v - m) / s if s > 1e-9 else 0.0) for v in values]


def winloss(rer: list[float], base: list[float]) -> dict[str, int]:
    w = sum(1 for a, b in zip(rer, base) if a < b - 1e-9)
    l = sum(1 for a, b in zip(rer, base) if a > b + 1e-9)
    return {"win": w, "loss": l, "tie": len(rer) - w - l}


# --------------------------------------------------------------------------- #
# numpy-only models                                                           #
# --------------------------------------------------------------------------- #
def _standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with an unregularized bias column."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    d = Xb.shape[1]
    reg = np.eye(d)
    reg[-1, -1] = 0.0  # do not regularize bias
    return np.linalg.solve(Xb.T @ Xb + lam * reg, Xb.T @ y)


def _ridge_pred(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ w


def _logreg_fit(X: np.ndarray, y: np.ndarray, lam: float, iters: int, lr: float) -> np.ndarray:
    """L2-regularized logistic regression via deterministic full-batch GD."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    n, d = Xb.shape
    w = np.zeros(d)
    reg = np.ones(d)
    reg[-1] = 0.0  # do not regularize bias
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30.0, 30.0)))
        grad = Xb.T @ (p - y) / n + lam * reg * w
        w -= lr * grad
    return w


def _logreg_pred(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.hstack([X, np.ones((X.shape[0], 1))]) @ w, -30.0, 30.0)))


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# data -> per-clip features                                                   #
# --------------------------------------------------------------------------- #
def build_clip_table(preds: list[dict[str, Any]], traj, time_step: float):
    """Group prediction rows by clip and compute per-candidate features/metrics."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        pk, gk = p.get("pred_xyz_npz_key"), p.get("gt_xyz_npz_key")
        if pk not in traj or gk not in traj:
            continue
        pred_xyz = np.asarray(traj[pk]).tolist()
        m = compute_trajectory_metrics(pred_xyz, np.asarray(traj[gk]).tolist(), time_step=time_step)
        groups.setdefault(p["sample_id"], []).append({
            "tid": p.get("trajectory_sample_id", 0), "ade": m["ade_m"], "fde": m["fde_m"],
            "xyz": pred_xyz, "feat": candidate_features(pred_xyz, time_step),
            "cot": p.get("cot"), "meta_action": p.get("meta_action"), "answer": p.get("answer"),
        })

    clips: list[dict[str, Any]] = []
    for sid, cands in groups.items():
        if len(cands) < 2:
            continue
        cands.sort(key=lambda c: c["tid"])
        first_idx = next((i for i, c in enumerate(cands) if c["tid"] == 0), 0)
        arrs = [np.asarray(c["xyz"], dtype=float)[:, :2] for c in cands]
        centroid = np.mean(arrs, axis=0)
        dist = [float(np.mean(np.linalg.norm(a - centroid, axis=1))) for a in arrs]
        mbr_idx = int(np.argmin(dist))

        cols = [zscore([c["feat"][k] for c in cands]) for k in FEAT_KEYS]
        cols.append(zscore(dist))                      # dist_to_consensus (the MBR signal)
        F = np.asarray(cols, dtype=float).T            # (n_cand, len(FEAT_KEYS)+1)

        ade = np.asarray([c["ade"] for c in cands], dtype=float)
        fde = np.asarray([c["fde"] for c in cands], dtype=float)
        oracle_idx = int(np.argmin(ade))
        is_best = np.zeros(len(cands)); is_best[oracle_idx] = 1.0
        intents = sorted(set(parse_intents(
            {"cot": cands[first_idx]["cot"], "meta_action": cands[first_idx]["meta_action"],
             "answer": cands[first_idx]["answer"]})))
        clips.append({
            "sample_id": sid, "tids": [c["tid"] for c in cands], "intents": intents,
            "F": F, "ade": ade, "fde": fde, "is_best": is_best, "tgt": np.asarray(zscore(list(ade))),
            "first_idx": first_idx, "mbr_idx": mbr_idx, "oracle_idx": oracle_idx,
        })
    return clips


# --------------------------------------------------------------------------- #
# leave-one-clip-out CV                                                       #
# --------------------------------------------------------------------------- #
def loo_select(clips, *, lam_ridge, lam_logreg, logreg_iters, logreg_lr):
    """For each clip, fit on all other clips and pick a candidate. Fills outcome keys."""
    X_all = np.vstack([c["F"] for c in clips])
    y_best = np.concatenate([c["is_best"] for c in clips])
    y_tgt = np.concatenate([c["tgt"] for c in clips])
    clip_ix = np.concatenate([np.full(len(c["ade"]), k) for k, c in enumerate(clips)])

    for k, c in enumerate(clips):
        train = clip_ix != k
        mu, sd = _standardize_fit(X_all[train])
        Xtr = (X_all[train] - mu) / sd
        Xte = (c["F"] - mu) / sd

        wr = _ridge_fit(Xtr, y_tgt[train], lam_ridge)
        ridge_idx = int(np.argmin(_ridge_pred(Xte, wr)))          # lower predicted ADE = better
        wl = _logreg_fit(Xtr, y_best[train], lam_logreg, logreg_iters, logreg_lr)
        logreg_idx = int(np.argmax(_logreg_pred(Xte, wl)))        # higher P(best) = better

        ade, fde = c["ade"], c["fde"]
        c.update({
            "ridge_idx": ridge_idx, "logreg_idx": logreg_idx,
            "first_ade": float(ade[c["first_idx"]]), "first_fde": float(fde[c["first_idx"]]),
            "mbr_ade": float(ade[c["mbr_idx"]]), "mbr_fde": float(fde[c["mbr_idx"]]),
            "ridge_ade": float(ade[ridge_idx]), "ridge_fde": float(fde[ridge_idx]),
            "logreg_ade": float(ade[logreg_idx]), "logreg_fde": float(fde[logreg_idx]),
            "oracle_ade": float(ade[c["oracle_idx"]]), "oracle_fde": float(fde[c["oracle_idx"]]),
            "rand_ade": float(ade.mean()),
        })
    return clips


def paired_bootstrap(clips, base_key, sel_key, n_boot, seed):
    """Paired bootstrap CI on mean(base_ade - sel_ade); positive = sel beats base."""
    if not clips or n_boot <= 0:
        return {}
    d_ade = np.asarray([c[base_key + "_ade"] - c[sel_key + "_ade"] for c in clips], dtype=float)
    d_fde = np.asarray([c[base_key + "_fde"] - c[sel_key + "_fde"] for c in clips], dtype=float)
    rng = np.random.default_rng(seed); n = len(clips)

    def ci(d):
        idx = rng.integers(0, n, size=(n_boot, n))
        means = d[idx].mean(axis=1)
        return {"mean_improve_m": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
                "frac_better": round(float((means > 0).mean()), 3)}

    return {"ade": ci(d_ade), "fde": ci(d_fde)}


def _closed(m_re, m_first, m_oracle):
    denom = m_first - m_oracle
    return round((m_first - m_re) / denom * 100, 2) if abs(denom) > 1e-9 else None


def analyze(clips):
    if not clips:
        return {"num_clips": 0}
    col = lambda k: [c[k] for c in clips]
    mfa, mff = _mean(col("first_ade")), _mean(col("first_fde"))
    moa, mof = _mean(col("oracle_ade")), _mean(col("oracle_fde"))
    mmbr, mridge, mlog = _mean(col("mbr_ade")), _mean(col("ridge_ade")), _mean(col("logreg_ade"))
    return {
        "num_clips": len(clips),
        "first_ade": round(mfa, 4), "first_fde": round(mff, 4),
        "mbr_ade": round(mmbr, 4), "mbr_fde": round(_mean(col("mbr_fde")), 4),
        "ridge_ade": round(mridge, 4), "ridge_fde": round(_mean(col("ridge_fde")), 4),
        "logreg_ade": round(mlog, 4), "logreg_fde": round(_mean(col("logreg_fde")), 4),
        "oracle_ade": round(moa, 4), "oracle_fde": round(mof, 4),
        "random_ade": round(_mean(col("rand_ade")), 4),
        "mbr_gap_closed_ade_pct": _closed(mmbr, mfa, moa),
        "ridge_gap_closed_ade_pct": _closed(mridge, mfa, moa),
        "logreg_gap_closed_ade_pct": _closed(mlog, mfa, moa),
        "ridge_vs_first": winloss(col("ridge_ade"), col("first_ade")),
        "logreg_vs_first": winloss(col("logreg_ade"), col("first_ade")),
        "ridge_vs_mbr": winloss(col("ridge_ade"), col("mbr_ade")),
        "logreg_vs_mbr": winloss(col("logreg_ade"), col("mbr_ade")),
    }


def verdict_from_ci(sig_vs_mbr: dict) -> str:
    if not sig_vs_mbr or "ade" not in sig_vs_mbr:
        return "no bootstrap (n_boot=0)"
    lo, hi = sig_vs_mbr["ade"]["ci95"]
    if lo > 0:
        return "learned BEATS MBR (95% CI on ADE excludes 0)"
    if hi < 0:
        return "learned LOSES to MBR (95% CI on ADE excludes 0)"
    return "learned ~= MBR (95% CI on ADE includes 0): no significant gain over free consensus"


def build_report(preds, traj, *, time_step=0.1, lam_ridge=1.0, lam_logreg=1.0,
                 logreg_iters=300, logreg_lr=0.5, n_boot=2000, seed=42,
                 run_name=None, split="val"):
    clips = build_clip_table(preds, traj, time_step)
    if not clips:
        return {"num_clips": 0, "note": "no usable clips"}, [], []
    loo_select(clips, lam_ridge=lam_ridge, lam_logreg=lam_logreg,
               logreg_iters=logreg_iters, logreg_lr=logreg_lr)

    sy = [c for c in clips if set(c["intents"]) & STOP_LIKE]
    sig_overall = {
        "mbr_vs_first": paired_bootstrap(clips, "first", "mbr", n_boot, seed),
        "ridge_vs_first": paired_bootstrap(clips, "first", "ridge", n_boot, seed),
        "logreg_vs_first": paired_bootstrap(clips, "first", "logreg", n_boot, seed),
        "ridge_vs_mbr": paired_bootstrap(clips, "mbr", "ridge", n_boot, seed),
        "logreg_vs_mbr": paired_bootstrap(clips, "mbr", "logreg", n_boot, seed),
    }
    sig_sy = {
        "ridge_vs_mbr": paired_bootstrap(sy, "mbr", "ridge", n_boot, seed),
        "logreg_vs_mbr": paired_bootstrap(sy, "mbr", "logreg", n_boot, seed),
    } if sy else {}

    selection_rows = [{
        "sample_id": c["sample_id"], "intents": c["intents"],
        "first_tid": c["tids"][c["first_idx"]], "mbr_tid": c["tids"][c["mbr_idx"]],
        "ridge_tid": c["tids"][c["ridge_idx"]], "logreg_tid": c["tids"][c["logreg_idx"]],
        "oracle_tid": c["tids"][c["oracle_idx"]],
        "first_ade": round(c["first_ade"], 3), "mbr_ade": round(c["mbr_ade"], 3),
        "ridge_ade": round(c["ridge_ade"], 3), "logreg_ade": round(c["logreg_ade"], 3),
        "oracle_ade": round(c["oracle_ade"], 3),
    } for c in clips]

    report = {
        "run_name": run_name, "split": split, "num_clips": len(clips),
        "features": list(FEAT_KEYS) + ["dist_to_consensus"],
        "hyperparams": {"lam_ridge": lam_ridge, "lam_logreg": lam_logreg,
                        "logreg_iters": logreg_iters, "logreg_lr": logreg_lr,
                        "bootstrap": n_boot, "seed": seed},
        "verdict_logreg_vs_mbr": verdict_from_ci(sig_overall["logreg_vs_mbr"]),
        "verdict_ridge_vs_mbr": verdict_from_ci(sig_overall["ridge_vs_mbr"]),
        "overall": analyze(clips), "stop_yield_subset": analyze(sy),
        "significance_overall": sig_overall, "significance_stop_yield": sig_sy,
    }
    return report, selection_rows, clips


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--run-name", default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--lam-ridge", type=float, default=1.0)
    p.add_argument("--lam-logreg", type=float, default=1.0)
    p.add_argument("--logreg-iters", type=int, default=300)
    p.add_argument("--logreg-lr", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for CIs (0 to skip).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--selection-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    preds = read_jsonl(a.predictions or bp["predictions"])
    traj = np.load(a.trajectories or bp["trajectories"])

    report, selection_rows, _ = build_report(
        preds, traj, time_step=a.time_step, lam_ridge=a.lam_ridge, lam_logreg=a.lam_logreg,
        logreg_iters=a.logreg_iters, logreg_lr=a.logreg_lr, n_boot=a.bootstrap, seed=a.seed,
        run_name=a.run_name, split=a.split)
    print(json.dumps(report, indent=2))

    out = a.out or (Path("outputs/runs") / a.run_name / "analysis" / "learned_rerank_report.json" if a.run_name else None)
    sel_out = a.selection_out or (Path("outputs/runs") / a.run_name / "analysis" / "learned_rerank_selection.jsonl" if a.run_name else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote {out}")
    if sel_out:
        sel_out.parent.mkdir(parents=True, exist_ok=True)
        sel_out.write_text("\n".join(json.dumps(r) for r in selection_rows) + "\n", encoding="utf-8")
        print(f"[ok] wrote {sel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
