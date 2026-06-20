#!/usr/bin/env python
"""Tier-3 set-aggregator reranker: a small permutation-invariant neural net that
scores the *set* of N candidates jointly, to test whether a learned aggregator can
beat the hand-coded MBR/consensus mean.

Why this and not "a bigger model on the same features"
------------------------------------------------------
B (scripts/06) already proved that a model free to weight the cheap per-candidate
features just reconstructs MBR (ridge ties it). Capacity was never the bottleneck --
*information* is. So a deeper net on the SAME 6 scalars would only reproduce MBR.

This script attacks MBR's actual weakness instead. MBR selects the candidate nearest
the **arithmetic mean** of the N sampled paths -- a crude aggregator that an outlier
or a bimodal split (turn-left vs turn-right) can fool. We give a DeepSets-style net
the raw down-sampled geometry of *all* N candidates and let it learn its own consensus
statistic (robust / mode-aware), rather than handing it the MBR distance. If a learned
set function beats MBR, the win is the better aggregator; if it ties, geometry is
genuinely exhausted and only richer (visual/internal) features can help (-> Tier-3
hidden-state verifier).

Design
------
- Per candidate input = [5 within-clip z-scored dynamics] + [K down-sampled waypoints
  relative to the path start]. We deliberately DO NOT feed dist-to-consensus; the
  set-pooling sees every candidate's geometry and must discover consensus itself.
- Model (numpy, backprop, Adam): phi(x_i) -> mean-pool over the set -> rho([phi_i, ctx])
  -> scalar score per candidate. Permutation-invariant by construction.
- Target = within-clip z-scored ADE (the same regression target ridge used); select
  argmin predicted score. GT is used only to form the target/label, never at selection.
- **k-fold clip-level CV** (not leave-one-out): a neural net is too slow to refit 1000x,
  so we train k models on held-out folds -- still an honest held-out estimate.
- Pure numpy / CPU on purpose (matches the repo's light analysis layer; sklearn/torch
  are not needed). Trivially portable to torch if you later want GPU/scale.

Run (after the baseline run produced outputs/runs/<run>/baseline/):
  python scripts/07_set_rerank.py --run-name val_cand5_n1000
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


# --- small helpers (kept self-contained, identical to 04/06) ------------------ #
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def candidate_features(xyz: list[list[float]], dt: float) -> dict[str, float]:
    pts = [(float(p[0]), float(p[1])) for p in xyz]
    if len(pts) < 2:
        return dict(final_speed=0.0, speed_delta=0.0, max_abs_jerk=0.0, heading_change=0.0, lateral_disp=0.0)
    speeds = [math.dist(pts[i], pts[i - 1]) / dt for i in range(1, len(pts))]
    final_speed = speeds[-1]
    speed_delta = speeds[0] - speeds[-1]
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


def paired_bootstrap(clips, base_key, sel_key, n_boot, seed):
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


def waypoints(xyz: np.ndarray, k: int) -> np.ndarray:
    """Down-sample an (T,>=2) path to k waypoints relative to its start; returns (2k,)."""
    xy = np.asarray(xyz, dtype=float)[:, :2]
    idx = np.linspace(0, len(xy) - 1, k).round().astype(int)
    rel = xy[idx] - xy[0]
    return rel.reshape(-1)


# --- DeepSets-style set regressor (numpy, Adam, permutation-invariant) -------- #
class SetRegressor:
    """phi (1 hidden) -> mean-pool over the set -> rho([phi_i, ctx]) (1 hidden) -> score.

    Inputs are shaped (C, N, D): C sets (clips), N candidates each, D features.
    Backprop is hand-derived and numerically gradient-checked in tests.
    """

    def __init__(self, d_in: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.standard_normal((a, b)) * math.sqrt(2.0 / a)
        self.P = {
            "W1": s(d_in, hidden), "b1": np.zeros(hidden),
            "W2": s(2 * hidden, hidden), "b2": np.zeros(hidden),
            "w3": s(hidden, 1)[:, 0] * 0.1, "b3": 0.0,
        }
        self.h = hidden

    def forward(self, X, cache=False):
        P = self.P
        A1 = np.einsum("cnd,dh->cnh", X, P["W1"]) + P["b1"]
        H1 = np.maximum(A1, 0.0)
        ctx = H1.mean(axis=1, keepdims=True)                       # C,1,H
        G = np.concatenate([H1, np.broadcast_to(ctx, H1.shape)], axis=2)
        A2 = np.einsum("cnk,kh->cnh", G, P["W2"]) + P["b2"]
        H2 = np.maximum(A2, 0.0)
        s = np.einsum("cnh,h->cn", H2, P["w3"]) + P["b3"]          # C,N
        if cache:
            self._c = (X, A1, H1, ctx, G, A2, H2)
        return s

    def backward(self, dS):
        X, A1, H1, ctx, G, A2, H2 = self._c
        P = self.P; N = X.shape[1]
        g = {}
        g["w3"] = np.einsum("cn,cnh->h", dS, H2)
        g["b3"] = float(dS.sum())
        dH2 = np.einsum("cn,h->cnh", dS, P["w3"])
        dA2 = dH2 * (A2 > 0)
        g["W2"] = np.einsum("cnk,cnh->kh", G, dA2)
        g["b2"] = dA2.sum(axis=(0, 1))
        dG = np.einsum("cnh,kh->cnk", dA2, P["W2"])
        H = self.h
        dH1 = dG[..., :H].copy()
        dctx = dG[..., H:].sum(axis=1)                             # C,H (ctx broadcast over N)
        dH1 += dctx[:, None, :] / N                                # mean-pool gradient
        dA1 = dH1 * (A1 > 0)
        g["W1"] = np.einsum("cnd,cnh->dh", X, dA1)
        g["b1"] = dA1.sum(axis=(0, 1))
        return g

    def fit(self, X, T, steps=400, lr=0.01, l2=1e-4, seed=0):
        """Full-batch Adam on MSE(score, target). T shaped (C,N)."""
        P = self.P
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        denom = T.size
        for t in range(1, steps + 1):
            s = self.forward(X, cache=True)
            dS = (2.0 / denom) * (s - T)
            g = self.backward(dS)
            for k in P:
                g[k] = g[k] + l2 * P[k]                            # L2
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * (g[k] * g[k])
                mhat = m[k] / (1 - b1 ** t); vhat = v[k] / (1 - b2 ** t)
                P[k] = P[k] - lr * mhat / (np.sqrt(vhat) + eps)
        return self

    def loss(self, X, T):
        s = self.forward(X)
        return float(((s - T) ** 2).mean())


# --- data -> per-clip set tensors -------------------------------------------- #
def build_clip_table(preds, traj, time_step, k_waypoints):
    groups: dict[str, list[dict[str, Any]]] = {}
    for p in preds:
        pk, gk = p.get("pred_xyz_npz_key"), p.get("gt_xyz_npz_key")
        if pk not in traj or gk not in traj:
            continue
        pred_xyz = np.asarray(traj[pk]).tolist()
        m = compute_trajectory_metrics(pred_xyz, np.asarray(traj[gk]).tolist(), time_step=time_step)
        groups.setdefault(p["sample_id"], []).append({
            "tid": p.get("trajectory_sample_id", 0), "ade": m["ade_m"], "fde": m["fde_m"],
            "xyz": np.asarray(traj[pk], dtype=float), "feat": candidate_features(pred_xyz, time_step),
            "cot": p.get("cot"), "meta_action": p.get("meta_action"), "answer": p.get("answer"),
        })
    clips = []
    n_cand = None
    for sid, cands in groups.items():
        if len(cands) < 2:
            continue
        cands.sort(key=lambda c: c["tid"])
        if n_cand is None:
            n_cand = len(cands)
        if len(cands) != n_cand:
            continue  # set net needs a constant N; real runs are 5/clip
        first_idx = next((i for i, c in enumerate(cands) if c["tid"] == 0), 0)
        arrs = [c["xyz"][:, :2] for c in cands]
        centroid = np.mean(arrs, axis=0)
        mbr_idx = int(np.argmin([float(np.mean(np.linalg.norm(a - centroid, axis=1))) for a in arrs]))
        dz = {key: zscore([cc["feat"][key] for cc in cands]) for key in FEAT_KEYS}
        dyn = list(zip(*[dz[key] for key in FEAT_KEYS]))      # N tuples, one per candidate
        X = np.array([
            list(dyn[i]) + list(waypoints(cands[i]["xyz"], k_waypoints))
            for i in range(len(cands))
        ], dtype=float)
        ade = np.array([c["ade"] for c in cands]); fde = np.array([c["fde"] for c in cands])
        oracle_idx = int(np.argmin(ade))
        intents = sorted(set(parse_intents(
            {"cot": cands[first_idx]["cot"], "meta_action": cands[first_idx]["meta_action"],
             "answer": cands[first_idx]["answer"]})))
        clips.append({"sample_id": sid, "tids": [c["tid"] for c in cands], "intents": intents,
                      "X": X, "ade": ade, "fde": fde, "tgt": np.asarray(zscore(list(ade))),
                      "first_idx": first_idx, "mbr_idx": mbr_idx, "oracle_idx": oracle_idx})
    return clips


def kfold_select(clips, *, hidden, steps, lr, kfolds, seed):
    """k-fold clip-level CV: train the set net on held-out folds, pick argmin score."""
    C = len(clips)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, kfolds, size=C)
    Xall = np.stack([c["X"] for c in clips])           # C,N,D
    Tall = np.stack([c["tgt"] for c in clips])         # C,N
    D = Xall.shape[2]
    setnn_idx = [0] * C
    for f in range(kfolds):
        tr = fold != f; te = fold == f
        if not te.any() or not tr.any():
            continue
        mu = Xall[tr].reshape(-1, D).mean(axis=0)
        sd = Xall[tr].reshape(-1, D).std(axis=0); sd[sd < 1e-9] = 1.0
        model = SetRegressor(D, hidden=hidden, seed=seed + f)
        model.fit((Xall[tr] - mu) / sd, Tall[tr], steps=steps, lr=lr, seed=seed + f)
        s = model.forward((Xall[te] - mu) / sd)        # |te|,N
        for j, ci in enumerate(np.where(te)[0]):
            setnn_idx[ci] = int(np.argmin(s[j]))
    for ci, c in enumerate(clips):
        idx = setnn_idx[ci]
        c.update({"setnn_idx": idx,
                  "first_ade": float(c["ade"][c["first_idx"]]), "first_fde": float(c["fde"][c["first_idx"]]),
                  "mbr_ade": float(c["ade"][c["mbr_idx"]]), "mbr_fde": float(c["fde"][c["mbr_idx"]]),
                  "setnn_ade": float(c["ade"][idx]), "setnn_fde": float(c["fde"][idx]),
                  "oracle_ade": float(c["ade"][c["oracle_idx"]]), "oracle_fde": float(c["fde"][c["oracle_idx"]]),
                  "rand_ade": float(c["ade"].mean())})
    return clips


def _closed(m_re, m_first, m_oracle):
    d = m_first - m_oracle
    return round((m_first - m_re) / d * 100, 2) if abs(d) > 1e-9 else None


def analyze(clips):
    if not clips:
        return {"num_clips": 0}
    col = lambda k: [c[k] for c in clips]
    mfa, moa = _mean(col("first_ade")), _mean(col("oracle_ade"))
    mmbr, mset = _mean(col("mbr_ade")), _mean(col("setnn_ade"))
    return {
        "num_clips": len(clips),
        "first_ade": round(mfa, 4), "first_fde": round(_mean(col("first_fde")), 4),
        "mbr_ade": round(mmbr, 4), "mbr_fde": round(_mean(col("mbr_fde")), 4),
        "setnn_ade": round(mset, 4), "setnn_fde": round(_mean(col("setnn_fde")), 4),
        "oracle_ade": round(moa, 4), "oracle_fde": round(_mean(col("oracle_fde")), 4),
        "random_ade": round(_mean(col("rand_ade")), 4),
        "mbr_gap_closed_ade_pct": _closed(mmbr, mfa, moa),
        "setnn_gap_closed_ade_pct": _closed(mset, mfa, moa),
        "setnn_vs_first": winloss(col("setnn_ade"), col("first_ade")),
        "setnn_vs_mbr": winloss(col("setnn_ade"), col("mbr_ade")),
    }


def verdict_from_ci(sig) -> str:
    if not sig or "ade" not in sig:
        return "no bootstrap (n_boot=0)"
    lo, hi = sig["ade"]["ci95"]
    if lo > 0:
        return "set-net BEATS MBR (95% CI on ADE excludes 0)"
    if hi < 0:
        return "set-net LOSES to MBR (95% CI on ADE excludes 0)"
    return "set-net ~= MBR (95% CI on ADE includes 0): learned aggregator does not beat the mean"


def build_report(preds, traj, *, time_step=0.1, k_waypoints=6, hidden=16, steps=400, lr=0.01,
                 kfolds=5, n_boot=2000, seed=42, run_name=None, split="val"):
    clips = build_clip_table(preds, traj, time_step, k_waypoints)
    if not clips:
        return {"num_clips": 0, "note": "no usable clips"}, [], []
    kfold_select(clips, hidden=hidden, steps=steps, lr=lr, kfolds=kfolds, seed=seed)
    sy = [c for c in clips if set(c["intents"]) & STOP_LIKE]
    sig_overall = {
        "setnn_vs_first": paired_bootstrap(clips, "first", "setnn", n_boot, seed),
        "mbr_vs_first": paired_bootstrap(clips, "first", "mbr", n_boot, seed),
        "setnn_vs_mbr": paired_bootstrap(clips, "mbr", "setnn", n_boot, seed),
    }
    sig_sy = {"setnn_vs_mbr": paired_bootstrap(sy, "mbr", "setnn", n_boot, seed)} if sy else {}
    selection = [{
        "sample_id": c["sample_id"], "intents": c["intents"],
        "first_tid": c["tids"][c["first_idx"]], "mbr_tid": c["tids"][c["mbr_idx"]],
        "setnn_tid": c["tids"][c["setnn_idx"]], "oracle_tid": c["tids"][c["oracle_idx"]],
        "first_ade": round(c["first_ade"], 3), "mbr_ade": round(c["mbr_ade"], 3),
        "setnn_ade": round(c["setnn_ade"], 3), "oracle_ade": round(c["oracle_ade"], 3),
    } for c in clips]
    report = {
        "run_name": run_name, "split": split, "num_clips": len(clips),
        "model": "DeepSets(numpy) phi->mean-pool->rho, regress within-clip z-ADE, argmin",
        "inputs": f"{len(FEAT_KEYS)} z-dynamics + {k_waypoints} waypoints(xy); dist-to-consensus deliberately excluded",
        "hyperparams": {"hidden": hidden, "steps": steps, "lr": lr, "kfolds": kfolds,
                        "k_waypoints": k_waypoints, "bootstrap": n_boot, "seed": seed},
        "verdict_setnn_vs_mbr": verdict_from_ci(sig_overall["setnn_vs_mbr"]),
        "overall": analyze(clips), "stop_yield_subset": analyze(sy),
        "significance_overall": sig_overall, "significance_stop_yield": sig_sy,
    }
    return report, selection, clips


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--run-name", default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--k-waypoints", type=int, default=6)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--selection-out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    preds = read_jsonl(a.predictions or bp["predictions"])
    traj = np.load(a.trajectories or bp["trajectories"])
    report, selection, _ = build_report(
        preds, traj, time_step=a.time_step, k_waypoints=a.k_waypoints, hidden=a.hidden,
        steps=a.steps, lr=a.lr, kfolds=a.kfolds, n_boot=a.bootstrap, seed=a.seed,
        run_name=a.run_name, split=a.split)
    print(json.dumps(report, indent=2))
    out = a.out or (Path("outputs/runs") / a.run_name / "analysis" / "set_rerank_report.json" if a.run_name else None)
    sel_out = a.selection_out or (Path("outputs/runs") / a.run_name / "analysis" / "set_rerank_selection.jsonl" if a.run_name else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote {out}")
    if sel_out:
        sel_out.parent.mkdir(parents=True, exist_ok=True)
        sel_out.write_text("\n".join(json.dumps(r) for r in selection) + "\n", encoding="utf-8")
        print(f"[ok] wrote {sel_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
