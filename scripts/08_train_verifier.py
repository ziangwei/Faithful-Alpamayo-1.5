#!/usr/bin/env python
"""Project 2.0 -- learned verifier head on the frozen VLM's hidden states.

The triangulated finding from A/B/C is that *geometry-only* selection is exhausted
at MBR (ridge ties it, logreg backfires, the set-net loses). The only remaining lever
is information the output (x,y) throws away: the model's internal scene understanding.
This script trains a tiny verifier head -- the driving analogue of best-of-N + a reward
model -- and answers one sharp question via an ablation:

    Does adding the frozen-VLM scene vector on top of the consensus/geometry features
    let a learned head beat free MBR?

Two heads, identical except for inputs:
  - "geom"        : dynamics + waypoints + dist-to-consensus      (should ~= MBR; reproduces B)
  - "geom+scene"  : the above ⊕ the clip's pooled VLM hidden state (the new information)
The **marginal** (geom+scene minus geom), with a paired bootstrap CI, is the clean
estimate of how much selection signal the hidden state carries beyond geometry.

Inputs (all from the existing run; the scene NPZ is the only new artifact):
  - predictions JSONL + trajectories NPZ  (as for 04/06/07)
  - scene vectors NPZ: key "{sample_id}__scene_vec" -> 1-D float vector per clip,
    produced by 02 with --dump-hidden (see docs/tier3_hidden_state_verifier.md).
If no scene NPZ is given, only the geom head runs (a sanity baseline; the scene
ablation is skipped with a clear note).

Model is a tiny numpy MLP (Adam, backprop gradient-checked in tests); k-fold clip-level
CV; regress within-clip z-ADE, select argmin. CPU. Portable to torch for scale.

Run (after dumping scene vectors on the server):
  python scripts/08_train_verifier.py --run-name val_cand5_n1000 \
      --scene outputs/runs/val_cand5_n1000/baseline/val_hidden.npz
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


# --- self-contained helpers (identical to 04/06/07) --------------------------- #
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


def winloss(rer, base):
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
        idx = rng.integers(0, n, size=(n_boot, n)); means = d[idx].mean(axis=1)
        return {"mean_improve_m": round(float(d.mean()), 4),
                "ci95": [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
                "frac_better": round(float((means > 0).mean()), 3)}

    return {"ade": ci(d_ade), "fde": ci(d_fde)}


def waypoints(xyz: np.ndarray, k: int) -> np.ndarray:
    xy = np.asarray(xyz, dtype=float)[:, :2]
    idx = np.linspace(0, len(xy) - 1, k).round().astype(int)
    return (xy[idx] - xy[0]).reshape(-1)


# --- tiny MLP (numpy, 1 hidden layer, Adam; backprop gradient-checked) -------- #
class MLP:
    def __init__(self, d_in: int, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.P = {
            "W1": rng.standard_normal((d_in, hidden)) * math.sqrt(2.0 / d_in), "b1": np.zeros(hidden),
            "w2": rng.standard_normal(hidden) * 0.1, "b2": 0.0,
        }

    def forward(self, X, cache=False):
        A1 = X @ self.P["W1"] + self.P["b1"]
        H1 = np.maximum(A1, 0.0)
        s = H1 @ self.P["w2"] + self.P["b2"]
        if cache:
            self._c = (X, A1, H1)
        return s

    def backward(self, dS):
        X, A1, H1 = self._c
        g = {"w2": H1.T @ dS, "b2": float(dS.sum())}
        dH1 = np.outer(dS, self.P["w2"])
        dA1 = dH1 * (A1 > 0)
        g["W1"] = X.T @ dA1
        g["b1"] = dA1.sum(axis=0)
        return g

    def fit(self, X, y, steps=400, lr=0.01, l2=1e-4):
        P = self.P
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(v) for k, v in P.items()}
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            s = self.forward(X, cache=True)
            dS = (2.0 / len(y)) * (s - y)
            g = self.backward(dS)
            for k in P:
                g[k] = g[k] + l2 * P[k]
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * (g[k] * g[k])
                P[k] = P[k] - lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v[k] / (1 - b2 ** t)) + eps)
        return self

    def loss(self, X, y):
        s = self.forward(X)
        return float(((s - y) ** 2).mean())


# --- data -> per-clip features ----------------------------------------------- #
def build_clip_table(preds, traj, scene, time_step, k_waypoints):
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
    clips, n_cand = [], None
    for sid, cands in groups.items():
        if len(cands) < 2:
            continue
        cands.sort(key=lambda c: c["tid"])
        if n_cand is None:
            n_cand = len(cands)
        if len(cands) != n_cand:
            continue
        first_idx = next((i for i, c in enumerate(cands) if c["tid"] == 0), 0)
        arrs = [c["xyz"][:, :2] for c in cands]
        centroid = np.mean(arrs, axis=0)
        dist = [float(np.mean(np.linalg.norm(a - centroid, axis=1))) for a in arrs]
        mbr_idx = int(np.argmin(dist))
        dz = {key: zscore([cc["feat"][key] for cc in cands]) for key in FEAT_KEYS}
        dyn = list(zip(*[dz[key] for key in FEAT_KEYS]))
        dist_z = zscore(dist)
        Xg = np.array([list(dyn[i]) + list(waypoints(cands[i]["xyz"], k_waypoints)) + [dist_z[i]]
                       for i in range(len(cands))], dtype=float)
        sc = scene.get(sid) if scene is not None else None
        scene_row = None
        if sc is not None:
            sc = np.asarray(sc, dtype=float)
            # accept a shared per-clip vector (1-D) or per-candidate features (2-D: N x H)
            if sc.ndim == 1 or (sc.ndim == 2 and sc.shape[0] == len(cands)):
                scene_row = sc
        ade = np.array([c["ade"] for c in cands]); fde = np.array([c["fde"] for c in cands])
        oracle_idx = int(np.argmin(ade))
        intents = sorted(set(parse_intents(
            {"cot": cands[first_idx]["cot"], "meta_action": cands[first_idx]["meta_action"],
             "answer": cands[first_idx]["answer"]})))
        clips.append({"sample_id": sid, "tids": [c["tid"] for c in cands], "intents": intents,
                      "Xg": Xg, "scene": scene_row, "ade": ade, "fde": fde,
                      "tgt": np.asarray(zscore(list(ade))), "first_idx": first_idx,
                      "mbr_idx": mbr_idx, "oracle_idx": oracle_idx})
    have_scene = bool(clips) and all(c["scene"] is not None for c in clips)
    return clips, have_scene


def _design(clips, use_scene):
    """Stack per-candidate rows; geom or geom+scene."""
    rows, tgt, clip_ix = [], [], []
    for k, c in enumerate(clips):
        Xg = c["Xg"]
        if use_scene:
            sc = c["scene"]
            S = sc if sc.ndim == 2 else np.broadcast_to(sc, (Xg.shape[0], sc.shape[0]))
            X = np.concatenate([Xg, S], axis=1)
        else:
            X = Xg
        rows.append(X); tgt.append(c["tgt"]); clip_ix.append(np.full(len(c["ade"]), k))
    return np.vstack(rows), np.concatenate(tgt), np.concatenate(clip_ix)


def apply_scene_pca(clips, k):
    """Unsupervised (label-free) PCA to shrink the scene features to k dims -- the fix for
    overfitting when the raw hidden state is huge (e.g. 4096-d) and clips are few. Fit once
    on all clips' scene rows; uses no labels, so there is no target leakage."""
    mats = [c["scene"][None, :] if c["scene"].ndim == 1 else c["scene"] for c in clips]
    X = np.vstack(mats)
    mu = X.mean(axis=0)
    comps = np.linalg.svd(X - mu, full_matrices=False)[2][:k]      # (k, D)
    for c in clips:
        c["scene"] = (c["scene"] - mu) @ comps.T
    return int(X.shape[1]), int(comps.shape[0])


def run_head(clips, use_scene, *, hidden, steps, lr, l2, kfolds, seed):
    """k-fold clip-level CV of the MLP head; returns per-clip selected index."""
    X, y, clip_ix = _design(clips, use_scene)
    C = len(clips)
    fold = np.random.default_rng(seed).integers(0, kfolds, size=C)
    sel = [0] * C
    for f in range(kfolds):
        tr_clip = np.where(fold != f)[0]; te_clip = np.where(fold == f)[0]
        if len(tr_clip) == 0 or len(te_clip) == 0:
            continue
        tr = np.isin(clip_ix, tr_clip)
        mu = X[tr].mean(axis=0); sd = X[tr].std(axis=0); sd[sd < 1e-9] = 1.0
        model = MLP(X.shape[1], hidden=hidden, seed=seed + f)
        model.fit((X[tr] - mu) / sd, y[tr], steps=steps, lr=lr, l2=l2)
        for ci in te_clip:
            rows = clip_ix == ci
            s = model.forward((X[rows] - mu) / sd)
            sel[ci] = int(np.argmin(s))
    return sel


def _closed(m_re, m_first, m_oracle):
    d = m_first - m_oracle
    return round((m_first - m_re) / d * 100, 2) if abs(d) > 1e-9 else None


def _fill(clips, prefix, sel):
    for ci, c in enumerate(clips):
        i = sel[ci]
        c[prefix + "_ade"] = float(c["ade"][i]); c[prefix + "_fde"] = float(c["fde"][i])
        c[prefix + "_idx"] = i


def _base(clips):
    for c in clips:
        c["first_ade"] = float(c["ade"][c["first_idx"]]); c["first_fde"] = float(c["fde"][c["first_idx"]])
        c["mbr_ade"] = float(c["ade"][c["mbr_idx"]]); c["mbr_fde"] = float(c["fde"][c["mbr_idx"]])
        c["oracle_ade"] = float(c["ade"][c["oracle_idx"]]); c["oracle_fde"] = float(c["fde"][c["oracle_idx"]])
        c["rand_ade"] = float(c["ade"].mean())


def analyze(clips, heads):
    if not clips:
        return {"num_clips": 0}
    col = lambda k: [c[k] for c in clips]
    mfa, moa = _mean(col("first_ade")), _mean(col("oracle_ade"))
    out = {"num_clips": len(clips),
           "first_ade": round(mfa, 4), "mbr_ade": round(_mean(col("mbr_ade")), 4),
           "oracle_ade": round(moa, 4), "random_ade": round(_mean(col("rand_ade")), 4),
           "mbr_gap_closed_ade_pct": _closed(_mean(col("mbr_ade")), mfa, moa)}
    for h in heads:
        out[f"{h}_ade"] = round(_mean(col(f"{h}_ade")), 4)
        out[f"{h}_gap_closed_ade_pct"] = _closed(_mean(col(f"{h}_ade")), mfa, moa)
        out[f"{h}_vs_mbr"] = winloss(col(f"{h}_ade"), col("mbr_ade"))
    if "geomscene" in heads and "geom" in heads:
        out["geomscene_vs_geom"] = winloss(col("geomscene_ade"), col("geom_ade"))
    return out


def verdict(sig, label_better, label_worse, label_tie):
    if not sig or "ade" not in sig:
        return "no bootstrap (n_boot=0)"
    lo, hi = sig["ade"]["ci95"]
    if lo > 0:
        return label_better
    if hi < 0:
        return label_worse
    return label_tie


def build_report(preds, traj, scene, *, time_step=0.1, k_waypoints=6, hidden=32, steps=400,
                 lr=0.01, l2=1e-3, kfolds=5, n_boot=2000, seed=42, scene_pca=0,
                 run_name=None, split="val"):
    clips, have_scene = build_clip_table(preds, traj, scene, time_step, k_waypoints)
    if not clips:
        return {"num_clips": 0, "note": "no usable clips"}, []
    _base(clips)
    scene_info = {"present": have_scene}
    if have_scene:
        scene_info["per_candidate"] = bool(clips[0]["scene"].ndim == 2)
        scene_info["dim"] = int(clips[0]["scene"].shape[-1])
        if scene_pca and scene_pca > 0 and scene_info["dim"] > scene_pca:
            orig, red = apply_scene_pca(clips, scene_pca)
            scene_info["pca"] = {"from": orig, "to": red}
            scene_info["dim"] = red
    heads = ["geom"]
    _fill(clips, "geom", run_head(clips, False, hidden=hidden, steps=steps, lr=lr, l2=l2, kfolds=kfolds, seed=seed))
    if have_scene:
        heads.append("geomscene")
        _fill(clips, "geomscene", run_head(clips, True, hidden=hidden, steps=steps, lr=lr, l2=l2, kfolds=kfolds, seed=seed))

    sy = [c for c in clips if set(c["intents"]) & STOP_LIKE]
    sig = {"geom_vs_mbr": paired_bootstrap(clips, "mbr", "geom", n_boot, seed)}
    if have_scene:
        sig["geomscene_vs_mbr"] = paired_bootstrap(clips, "mbr", "geomscene", n_boot, seed)
        sig["geomscene_vs_geom"] = paired_bootstrap(clips, "geom", "geomscene", n_boot, seed)  # the marginal of scene
    report = {
        "run_name": run_name, "split": split, "num_clips": len(clips),
        "scene_vectors_present": have_scene,
        "scene": scene_info,
        "ablation": "geom (dynamics+waypoints+dist_to_consensus) vs geom+scene (⊕ pooled VLM hidden state)",
        "hyperparams": {"hidden": hidden, "steps": steps, "lr": lr, "l2": l2, "kfolds": kfolds,
                        "k_waypoints": k_waypoints, "bootstrap": n_boot, "seed": seed},
        "overall": analyze(clips, heads), "stop_yield_subset": analyze(sy, heads),
        "significance_overall": sig,
        "significance_stop_yield": ({"geomscene_vs_mbr": paired_bootstrap(sy, "mbr", "geomscene", n_boot, seed)}
                                    if have_scene and sy else {}),
    }
    if have_scene:
        report["verdict_geomscene_vs_mbr"] = verdict(
            sig["geomscene_vs_mbr"], "verifier BEATS MBR (CI excludes 0)",
            "verifier LOSES to MBR (CI excludes 0)", "verifier ~= MBR (CI includes 0)")
        report["verdict_scene_marginal"] = verdict(
            sig["geomscene_vs_geom"], "scene ADDS selection signal beyond geometry (CI excludes 0)",
            "scene HURTS vs geometry-only (CI excludes 0)",
            "scene adds NO signal beyond geometry (CI includes 0)")
    else:
        report["note"] = "no scene vectors -> only geom head ran (sanity baseline). Dump hidden states (02 --dump-hidden) for the ablation."
    return report, clips


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="val")
    p.add_argument("--run-name", default=None)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trajectories", type=Path, default=None)
    p.add_argument("--scene", type=Path, default=None,
                   help="NPZ of hidden features: '{sid}__scene_vec' (shared 1-D) or '{sid}__expert_hidden' (per-candidate N x H).")
    p.add_argument("--scene-pca", type=int, default=0,
                   help="Shrink scene features to this many PCA dims before training (0=off). Fixes overfitting on huge hidden states.")
    p.add_argument("--time-step", type=float, default=0.1)
    p.add_argument("--k-waypoints", type=int, default=6)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--kfolds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _load_scene(path):
    """Load per-clip features. Accepts a shared 1-D vector ('{sid}__scene_vec') or a
    per-candidate 2-D array N x H ('{sid}__expert_hidden'); the 2-D form is kept as-is."""
    if path is None:
        return None
    npz = np.load(path)
    out = {}
    for k in npz.files:
        for suf in ("__scene_vec", "__expert_hidden"):
            if k.endswith(suf):
                arr = np.asarray(npz[k], dtype=float)
                out[k[: -len(suf)]] = arr.reshape(-1) if arr.ndim == 1 else arr
                break
    return out or None


def main() -> int:
    a = parse_args()
    bp = baseline_output_paths(split=a.split, run_name=a.run_name)
    preds = read_jsonl(a.predictions or bp["predictions"])
    traj = np.load(a.trajectories or bp["trajectories"])
    scene = _load_scene(a.scene)
    report, _ = build_report(preds, traj, scene, time_step=a.time_step, k_waypoints=a.k_waypoints,
                             hidden=a.hidden, steps=a.steps, lr=a.lr, l2=a.l2, kfolds=a.kfolds,
                             n_boot=a.bootstrap, seed=a.seed, scene_pca=a.scene_pca,
                             run_name=a.run_name, split=a.split)
    print(json.dumps(report, indent=2))
    out = a.out or (Path("outputs/runs") / a.run_name / "analysis" / "verifier_report.json" if a.run_name else None)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
